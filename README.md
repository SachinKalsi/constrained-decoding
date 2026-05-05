# constrained-decoding

Trie-based constrained decoding for LLM classification.

Forces any open-weight model to output **exactly** one of your taxonomy labels — structural guarantee, not a probabilistic one. Supports single-label and multi-label classification with automatic repeat-prevention.

Read the full explanation: [Constrained Decoding: Forcing LLMs to Respect Your Taxonomy](https://medium.com/@sachinkalsi/constrained-decoding-forcing-llms-to-respect-your-taxonomy-3aaaf13329f9)

---

## How it works

At every decode step, a `LogitsProcessor` intercepts the logit vector and sets every invalid token to `-inf`. Since `exp(-inf) = 0`, those tokens get zero probability under softmax regardless of the sampling strategy. A trie built from your tokenized label set determines which tokens are valid at each step.

See the blog post for the full walkthrough, diagrams, and correctness proof.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Quick start

### Single-label

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from constrained_decoding import ConstrainedTrie, TrieLogitsProcessor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

labels = ["Science", "Sports", "Politics", "Technology"]
trie = ConstrainedTrie()
for label in labels:
    trie.insert(tokenizer.encode(" " + label, add_special_tokens=False))

trie.verify(labels, tokenizer)   # one-time build-time proof

prompt = "Classify: 'The match ended in a penalty shootout.'\nCategory:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

processor = TrieLogitsProcessor(trie, input_ids.shape[1], tokenizer.eos_token_id)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
output = model.generate(input_ids, logits_processor=LogitsProcessorList([processor]))
print(tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip())
# → "Sports"  (guaranteed to be in labels)
```

### Multi-label (no repeats)

```python
from constrained_decoding import MultiLabelTrieLogitsProcessor

sep_ids = tokenizer.encode(", ", add_special_tokens=False)

processor = MultiLabelTrieLogitsProcessor(
    trie=trie,
    prompt_length=input_ids.shape[1],
    eos_token_id=tokenizer.eos_token_id,
    sep_token_ids=sep_ids,
)
```

Full examples: [`examples/single_label.py`](examples/single_label.py) · [`examples/multi_label.py`](examples/multi_label.py)

---

## Key constraints

- Requires access to model logits — does not work with closed-API models (OpenAI, Anthropic, Gemini).
- Rebuild the trie whenever the label set changes.
- Does not improve model accuracy — it enforces output structure only.

---

## File structure

```
constrained_decoding/
├── trie.py          # TrieNode, ConstrainedTrie
└── processors.py    # TrieLogitsProcessor, MultiLabelTrieLogitsProcessor

examples/
├── single_label.py
└── multi_label.py
```
