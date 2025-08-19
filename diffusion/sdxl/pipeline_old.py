# sdxl.py

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler

from PIL import Image


class SDXLPipeline(nn.Module):
    def __init__(self, 
                 vae: AutoencoderKL, 
                 text_encoders: nn.ModuleList, 
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel, 
                 scheduler: DPMSolverMultistepScheduler):
        super().__init__()
        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        torch_dtype: torch.dtype = torch.float16,
                        device: str = "cuda"):
        """
        Builds every component from the Hugging Face checkpoint and
        loads its weights so that everything lines up correctly.
        """
        # 1) Tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # 2) VAE
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch_dtype
        ).to(device)

        # 3) Two CLIP text encoders (base + refiner)
        te1 = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype
        ).to(device)
        te2 = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype
        ).to(device)

        # 4) UNet
        unet : UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=torch_dtype
        ).to(device).eval()

        # 5) Scheduler
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler"
        )

        return cls(
            vae=vae,
            text_encoders=nn.ModuleList([te1, te2]),
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )

    @torch.no_grad()
    def __call__(self,
                prompt: str,
                num_inference_steps: int = 30,
                guidance_scale: float = 6.0,
                height: int = 1024,
                width: int = 1024):
        # 1) Tokenize & encode
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                max_length=self.tokenizer.model_max_length).to(self.unet.device)
        # first CLIP
        out1 = self.text_encoders[0](**inputs, output_hidden_states=True)
        # pooled = out1.hidden_states[-1][:,0] if you want CLS‑pooling,
        # or use the projection head on your second encoder for pooling.
        pooled1 = out1.pooler_output
        # second CLIP (the “refiner”)
        out2 = self.text_encoders[1](**inputs, output_hidden_states=True)
        pooled2 = out2.pooler_output

        # cat for full pooled prompt
        pooled_prompt_embeds = torch.cat([pooled1, pooled2], dim=-1)        # [B, C_pool]
        # now your per‑token embeddings for cross‑attention:
        txt_emb1 = out1.hidden_states[-2]                                   # [B, S, C1]
        txt_emb2 = out2.hidden_states[-2]                                   # [B, S, C2]
        prompt_embeds = torch.cat([txt_emb1, txt_emb2], dim=-1)             # [B, S, C_text]

        batch_size = prompt_embeds.shape[0]

        # 2) Prepare your micro‑conditioned IDs exactly like the pipeline:
        #    give it (orig_size, crop, target), no cropping =((h,w),(0,0),(h,w))
        text_enc_proj_dim = self.text_encoders[1].config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_enc_proj_dim,
        )                                                                   # [1, D]
        # if you do unconditional guidance, you’d also compute negative_add_time_ids here
        # tile to batch:
        add_text_embeds = pooled_prompt_embeds                                # [B, C_pool]
        add_time_ids   = add_time_ids.to(self.unet.device).repeat(batch_size, 1)  # [B, D]

        # 3) Prepare latents
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            device=self.unet.device,
            dtype=self.unet.dtype
        )
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, self.unet.device)

        # 4) Denoising
        for t in timesteps:
            # scale + guidance split if you like…
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds,   # [B, C_pool]
                    "time_ids":    add_time_ids      # [B, D]
                },
            ).sample

            latents = self.scheduler.step(
                noise_pred, 
                t, 
                latents, 
                guidance_scale=guidance_scale
            ).prev_sample

        # 5) Decode
        image = self.vae.decode(latents / 0.18215).sample
        return (image.clamp(-1,1) + 1) / 2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device.startswith("cuda") else torch.float32

def main():
    # 1) Build from the Hugging Face checkpoint
    print("Loading model..")
    pipe = SDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        device=device
    )

    # 2) Generate
    print("Generating image")
    image_tensor = pipe("A warrior goddess in a storm, epic lighting")
    # image_tensor shape: (1, 3, H, W) in [0,1]

    # 3) Convert to PIL and save
    print("Convert Image")
    img = (image_tensor[0].permute(1,2,0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(img).save("sdxl_out.png")


if __name__ == '__main__':
    main()