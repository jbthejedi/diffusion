# sdxl_scratch.py
import torch
import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DPMSolverMultistepScheduler

from PIL import Image
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1) Helpers pulled straight (and lightly trimmed) from
#    diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_timesteps(scheduler, num_inference_steps, device):
    scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps, num_inference_steps

def get_add_time_ids(
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    dtype: torch.dtype,
    text_encoder_projection_dim: int,
    unet: UNet2DConditionModel,
) -> torch.Tensor:
    """
    Builds the "time_ids" vector for micro-conditioning, matching
the UNet's add_embedding.linear_1.in_features.
    """
    # 1) assemble components
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # 2) compute dims
    passed_dim = unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    expected_dim = unet.add_embedding.linear_1.in_features
    if passed_dim != expected_dim:
        raise ValueError(f"expected add_embed_dim={expected_dim} but built {passed_dim}")

    # 3) return tensor
    return torch.tensor([add_time_ids], dtype=dtype)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Build your “from_pretrained” once, exactly as you already have:
# ─────────────────────────────────────────────────────────────────────────────

class SDXLFromScratch(torch.nn.Module):
    def __init__(self, vae, text_encoders, tokenizer, unet, scheduler):
        super().__init__()
        self.vae, self.text_encoders, self.tokenizer = vae, text_encoders, tokenizer
        self.unet, self.scheduler = unet, scheduler

    @classmethod
    def from_pretrained(cls, repo, torch_dtype, device):
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=torch_dtype).to(device)
        te1 = CLIPTextModel .from_pretrained(repo, subfolder="text_encoder",   torch_dtype=torch_dtype).to(device)
        # te2 = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch_dtype).to(device)
        te2 = CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch_dtype).to(device)
        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=torch_dtype).to(device).eval()
        sched = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
        return cls(vae, [te1, te2], tok, unet, sched)

    @torch.no_grad()
    def forward(self, prompt: str,
                num_inference_steps=30, guidance_scale=6.0,
                height=1024, width=1024):
        device = self.unet.device
        # — Tokenize & encode —
        tok_out = self.tokenizer(prompt,
                                 return_tensors="pt",
                                 padding="max_length",
                                 max_length=self.tokenizer.model_max_length,
                                 truncation=True).to(device)
        # first text encoder
        out1 = self.text_encoders[0](**tok_out, output_hidden_states=True)
        pooled1 = out1.pooler_output          # [B, Cproj1]
        # second text encoder (the “refiner”)
        out2 = self.text_encoders[1](**tok_out, output_hidden_states=True)
        pooled2 = out2.text_embeds # [B, Cproj2]
        # Combine
        pooled_prompt = torch.cat([pooled1, pooled2], dim=-1)   # [B, C_pool]

        # per‑token embeddings for cross‑attention
        txt_emb1 = out1.hidden_states[-2]     # [B, S, C1]
        txt_emb2 = out2.hidden_states[-2]     # [B, S, C2]
        prompt_embeds = torch.cat([txt_emb1, txt_emb2], dim=-1)  # [B, S, C_text]

        # add_text = pooled_prompt.to(device)                    # <— define this!
        add_text = pooled2.to(device)                    # <— define this!

        # text_encoder_projection_dim = (
        #     int(pooled_prompt.shape[-1])
        #     if self.text_encoders[1] is None
        #     else self.text_encoders[1].config.projection_dim
        # )
        text_encoder_projection_dim = self.text_encoders[1].config.projection_dim  # 1280


        bs = prompt_embeds.shape[0]
        # exactly the same helper logic as HuggingFace’s
        add_time = get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            dtype=prompt_embeds.dtype,
            # text_encoder_projection_dim=self.text_encoders[1].config.projection_dim,
            text_encoder_projection_dim=text_encoder_projection_dim,
            unet=self.unet,
        ).to(device).repeat(bs, 1)  # [B, D]

        # if self.do_classifier_free_guidance:
        #     add_time = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # add_text_embeds = add_text_embeds.repeat(batch_size * num_images_per_prompt, 1)
        # add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)


        # — latents + timesteps —
        latents = torch.randn(
            (bs, self.unet.config.in_channels, height // 8, width // 8),
            device=device, dtype=self.unet.dtype
        )
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        # — denoising loop —
        for t in tqdm.tqdm(timesteps):
            noise_pred = self.unet(
                latents, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": add_text,
                    "time_ids":    add_time,
                }
            ).sample

            latents = self.scheduler.step(
                noise_pred, t, latents,
                # guidance_scale=guidance_scale
            ).prev_sample

        # — decode & return —
        img = self.vae.decode(latents / 0.18215).sample
        return (img.clamp(-1,1) + 1) / 2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device.startswith("cuda") else torch.float32

def main():
    # 1) Build from the Hugging Face checkpoint
    print("Loading model..")
    repo = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = SDXLFromScratch.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        device=device,
    )

    # 2) Generate
    print("Generating image")
    image_tensor = pipe(
        "A warrior goddess in a storm, epic lighting",
        num_inference_steps=5,
    )
    # image_tensor shape: (1, 3, H, W) in [0,1]

    # 3) Convert to PIL and save
    print("Convert Image")
    img = (image_tensor[0].permute(1,2,0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(img).save("sdxl_out.png")


if __name__ == '__main__':
    main()
