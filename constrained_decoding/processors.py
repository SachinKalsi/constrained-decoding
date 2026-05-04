"""
processors.py — HuggingFace LogitsProcessor implementations.

Both processors intercept the logit vector at every decode step and set
all invalid token logits to -inf before sampling. Since exp(-inf) = 0,
those tokens get zero probability under any sampling strategy.
"""

import torch
from transformers import LogitsProcessor

from .trie import ConstrainedTrie


class TrieLogitsProcessor(LogitsProcessor):
    """
    Single-label constrained decoding.

    At each step, masks every token that is not a valid continuation of
    the label being generated. When the trie pointer reaches an end node,
    EOS is added so generation can terminate.

    Parameters
    ----------
    trie          : ConstrainedTrie built from your label set.
    prompt_length : Number of tokens in the prompt (before generation starts).
    eos_token_id  : Token ID for end-of-sequence.
    """

    def __init__(self, trie, prompt_length, eos_token_id):
        self.trie          = trie
        self.prompt_length = prompt_length
        self.eos           = eos_token_id

    def __call__(self, input_ids, scores):
        # what has the model generated so far (after the prompt)?
        generated = input_ids[0, self.prompt_length:].tolist()

        # ask the trie: which tokens are valid at this point?
        valid = self.trie.get_valid_next_tokens(generated)

        # if we've reached a complete label, the model can also stop
        if self.trie.is_complete(generated):
            valid.add(self.eos)

        # set every other token to -inf so it cannot be sampled
        masked = torch.full_like(scores, float("-inf"))
        for tid in valid:
            masked[0, tid] = scores[0, tid]
        return masked


class MultiLabelTrieLogitsProcessor(LogitsProcessor):
    """
    Multi-label constrained decoding.

    After any complete label the model can either stop (EOS) or continue
    with a separator token. If it continues, already-emitted labels are
    excluded at ROOT so the same label cannot be picked again.

    Parameters
    ----------
    trie          : ConstrainedTrie built from your label set.
    prompt_length : Number of tokens in the prompt.
    eos_token_id  : Token ID for end-of-sequence.
    sep_token_ids : Token IDs for the separator (e.g. tokenizer.encode(", ", add_special_tokens=False)).
    """

    def __init__(self, trie, prompt_length, eos_token_id, sep_token_ids):
        self.trie          = trie
        self.prompt_length = prompt_length
        self.eos           = eos_token_id
        self.sep           = sep_token_ids
        # build once at init: first token -> all labels that start with it.
        # used at ROOT to check which first tokens are fully exhausted.
        self._by_first = {}
        for label in trie.all_labels():
            self._by_first.setdefault(label[0], []).append(label)

    def _parse(self, tokens):
        """Split token list on separator -> (seen labels, current partial label)."""
        seen, current, sep_len = [], [], len(self.sep)
        i = 0
        while i < len(tokens):
            if tokens[i : i + sep_len] == self.sep:
                if current:
                    seen.append(tuple(current))
                current, i = [], i + sep_len
            else:
                current.append(tokens[i])
                i += 1
        return set(seen), current

    def __call__(self, input_ids, scores):
        generated     = input_ids[0, self.prompt_length:].tolist()
        seen, current = self._parse(generated)

        valid = self.trie.get_valid_next_tokens(current)

        if self.trie.is_complete(current):
            valid.add(self.eos)       # model can stop ...
            valid.update(self.sep)    # ... or continue with a separator

        # back at root after a separator: remove a first token only when
        # every label that starts with it has been seen. this ensures
        # sibling labels (e.g. "Technology/AI" after "Technology") are
        # never accidentally blocked.
        if not current and seen:
            for first_tok, group in self._by_first.items():
                if all(lbl in seen for lbl in group):
                    valid.discard(first_tok)
            if not valid:
                valid.add(self.eos)   # safety: no new labels available

        masked = torch.full_like(scores, float("-inf"))
        for tid in valid:
            masked[0, tid] = scores[0, tid]
        return masked
