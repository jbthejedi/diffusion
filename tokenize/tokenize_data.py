from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

files = [ str(Path("flickr30k/captions.txt")) ]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=files,
    vocab_size=10_000,       # pick a size (5k–10k is plenty for Flickr30k)
    min_frequency=2,         # only keep subwords that appear ≥2×
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# 3️⃣ Save your new vocab & merges
tokenizer.save_model("flickr30k_tokenizer")  
# → Creates `flickr30k_tokenizer-vocab.json` & `flickr30k_tokenizer-merges.txt`

# 4️⃣ Wrap with HuggingFace PreTrainedTokenizerFast for easy batching
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="flickr30k_tokenizer-tokenizer.json",  # or use vocab/merges directly
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

# 5️⃣ Test it out
sample = "A dog playing frisbee by the lake."
enc = hf_tokenizer(sample, padding="max_length", truncation=True, max_length=77)
print(enc.input_ids)
print(hf_tokenizer.convert_ids_to_tokens(enc.input_ids))
