"""
Single-label classification with trie-constrained decoding.

Output is guaranteed to be exactly one of the provided labels,
regardless of model weights or sampling strategy.

Usage
-----
python examples/single_label.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from constrained_decoding import ConstrainedTrie, TrieLogitsProcessor


# ── 1. Define your taxonomy ────────────────────────────────────────────────────

LABELS = ["Science", "Sports", "Politics", "Technology"]

# ── 2. Build the trie ──────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

trie = ConstrainedTrie()
for label in LABELS:
    # always tokenize as continuation (leading space), not as sentence start
    token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
    trie.insert(token_ids)

# sanity check: verify every label round-trips correctly
trie.verify(LABELS, tokenizer)
print("Trie verified.")

# ── 3. Build the prompt ────────────────────────────────────────────────────────

text = "Scientists discovered a new species of deep-sea fish near hydrothermal vents."
label_list = ", ".join(LABELS)

prompt = (
    f"Classify the text into exactly one of these categories: {label_list}.\n"
    f"Text: {text}\n"
    f"Category:"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
prompt_length = input_ids.shape[1]

# ── 4. Load the model and run constrained generation ──────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = TrieLogitsProcessor(
    trie=trie,
    prompt_length=prompt_length,
    eos_token_id=tokenizer.eos_token_id,
)

with torch.no_grad():
    output = model.generate(
        input_ids.to(model.device),
        logits_processor=LogitsProcessorList([processor]),
        max_new_tokens=20,
        do_sample=False,   # greedy; swap to True + temperature for sampling
    )

label = tokenizer.decode(
    output[0, prompt_length:], skip_special_tokens=True
).strip()

print(f"Input : {text}")
print(f"Output: {label}")
assert label in LABELS, f"Constraint violated: '{label}' is not in the label set"
print("Constraint check passed.")
