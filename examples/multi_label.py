"""
Multi-label classification with trie-constrained decoding.

Output is a comma-separated list of labels, each guaranteed to be from
the provided taxonomy, with no repeats.  The model stops when it emits
EOS or when all labels have been used.

Usage
-----
python examples/multi_label.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from constrained_decoding import ConstrainedTrie, MultiLabelTrieLogitsProcessor


# ── 1. Define your taxonomy ────────────────────────────────────────────────────

LABELS = ["Science", "Sports", "Politics", "Technology"]

# ── 2. Build the trie ──────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

trie = ConstrainedTrie()
for label in LABELS:
    token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
    trie.insert(token_ids)

trie.verify(LABELS, tokenizer)
print("Trie verified.")

# separator: ", " between labels
sep_token_ids = tokenizer.encode(", ", add_special_tokens=False)
print(f"Separator token IDs: {sep_token_ids}  "
      f"(decoded: '{tokenizer.decode(sep_token_ids)}')")

# ── 3. Build the prompt ────────────────────────────────────────────────────────

text = (
    "The government announced new funding for climate research "
    "while the national football team secured a spot in the finals."
)
label_list = ", ".join(LABELS)

prompt = (
    f"Classify the text into one or more of these categories: {label_list}.\n"
    f"Separate multiple categories with a comma.\n"
    f"Text: {text}\n"
    f"Categories:"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
prompt_length = input_ids.shape[1]

# ── 4. Load model and run constrained generation ──────────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = MultiLabelTrieLogitsProcessor(
    trie=trie,
    prompt_length=prompt_length,
    eos_token_id=tokenizer.eos_token_id,
    sep_token_ids=sep_token_ids,
)

with torch.no_grad():
    output = model.generate(
        input_ids.to(model.device),
        logits_processor=LogitsProcessorList([processor]),
        max_new_tokens=64,
        do_sample=False,
    )

raw = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()
predicted_labels = [l.strip() for l in raw.split(",") if l.strip()]

print(f"\nInput  : {text}")
print(f"Raw output : {raw}")
print(f"Labels     : {predicted_labels}")

# verify no label outside taxonomy, no duplicates
assert all(l in LABELS for l in predicted_labels), \
    f"Constraint violated: unexpected label in {predicted_labels}"
assert len(predicted_labels) == len(set(predicted_labels)), \
    f"Constraint violated: duplicate labels in {predicted_labels}"

print("Constraint checks passed.")
