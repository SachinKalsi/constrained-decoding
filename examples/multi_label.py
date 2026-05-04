"""
Multi-label classification with trie-constrained decoding.

The output is a comma-separated list of labels. Each label is guaranteed
to be from the taxonomy, with no repeats. The model stops whenever it
picks EOS — it is not forced to emit all labels.

Run: python examples/multi_label.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from constrained_decoding import ConstrainedTrie, MultiLabelTrieLogitsProcessor

# 1. labels
LABELS = ["Science", "Sports", "Politics", "Technology"]

# 2. build the trie
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

trie = ConstrainedTrie()
for label in LABELS:
    trie.insert(tokenizer.encode(" " + label, add_special_tokens=False))

trie.verify(LABELS, tokenizer)
print("Trie built and verified.")

sep_ids = tokenizer.encode(", ", add_special_tokens=False)
print(f"Separator: {sep_ids} → '{tokenizer.decode(sep_ids)}'")

# 3. prompt
text = (
    "The government announced new funding for climate research "
    "while the national football team secured a spot in the finals."
)

prompt = (
    f"Classify the text into one or more of: {', '.join(LABELS)}.\n"
    f"Separate multiple labels with a comma.\n"
    f"Text: {text}\n"
    f"Labels:"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 4. constrained generation
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = MultiLabelTrieLogitsProcessor(
    trie, input_ids.shape[1], tokenizer.eos_token_id, sep_ids
)

with torch.no_grad():
    output = model.generate(
        input_ids.to(model.device),
        logits_processor=LogitsProcessorList([processor]),
        max_new_tokens=64,
        do_sample=False,
    )

raw    = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
labels = [lb.strip() for lb in raw.split(",") if lb.strip()]

print(f"\nInput  : {text}")
print(f"Output : {raw}")
print(f"Labels : {labels}")

assert all(lb in LABELS for lb in labels), f"Unexpected label in: {labels}"
assert len(labels) == len(set(labels)),    f"Duplicate labels in: {labels}"
print("Constraint checks passed.")
