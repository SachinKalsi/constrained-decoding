"""
Single-label classification with trie-constrained decoding.

The output is guaranteed to be exactly one of the provided labels,
regardless of model weights or sampling strategy.

Run: python examples/single_label.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from constrained_decoding import ConstrainedTrie, TrieLogitsProcessor

# 1. labels
LABELS = ["Science", "Sports", "Politics", "Technology"]

# 2. build the trie
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

trie = ConstrainedTrie()
for label in LABELS:
    # tokenize as a continuation (leading space), not as a sentence start
    trie.insert(tokenizer.encode(" " + label, add_special_tokens=False))

trie.verify(LABELS, tokenizer)  # one-time build-time check
print("Trie built and verified.")

# 3. prompt
text = "Scientists discovered a new species of deep-sea fish near hydrothermal vents."

prompt = (
    f"Classify the text into one of: {', '.join(LABELS)}.\n"
    f"Text: {text}\n"
    f"Category:"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 4. constrained generation
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = TrieLogitsProcessor(trie, input_ids.shape[1], tokenizer.eos_token_id)

with torch.no_grad():
    output = model.generate(
        input_ids.to(model.device),
        logits_processor=LogitsProcessorList([processor]),
        max_new_tokens=20,
        do_sample=False,
    )

label = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

print(f"Input : {text}")
print(f"Label : {label}")
assert label in LABELS, f"Constraint violated: '{label}' not in label set"
