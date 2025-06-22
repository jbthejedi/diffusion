from pathlib import Path
import csv
import re

unique_words = set()
total_tokens = 0

# adjust this path to wherever you saved captions.txt
path = "/Users/justinbarry/projects/flickr30k_entities/flickr30k/captions.txt"

with Path(path).open(newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip the "image,caption" header
    for image, caption in reader:
        # lowercase + extract word characters (so "yard." → "yard")
        tokens = re.findall(r"\w+", caption.lower())
        total_tokens += len(tokens)
        unique_words.update(tokens)

print(f"Total word tokens:   {total_tokens}")
print(f"Unique word types:   {len(unique_words)}")
