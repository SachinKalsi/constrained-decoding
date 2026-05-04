"""
processors.py
-------------
HuggingFace LogitsProcessor implementations for trie-constrained decoding.

Both processors intercept the logit vector at every decode step and set
all invalid token logits to -inf before sampling.  The sampling strategy
(greedy, temperature, top-p, etc.) does not matter — invalid tokens have
probability exactly 0 under any strategy.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from .trie import ConstrainedTrie, TrieNode


class TrieLogitsProcessor(LogitsProcessor):
    """
    Single-label constrained decoding.

    At each step, masks every token that is not a valid continuation of
    the label being generated.  When the trie pointer reaches an end node,
    EOS is added to valid tokens so generation can terminate.

    Parameters
    ----------
    trie          : ConstrainedTrie built from your label set.
    prompt_length : Number of tokens in the prompt (input_ids length before generation).
    eos_token_id  : Token ID for end-of-sequence.
    """

    def __init__(
        self,
        trie: ConstrainedTrie,
        prompt_length: int,
        eos_token_id: int,
    ) -> None:
        self.trie = trie
        self.prompt_length = prompt_length
        self.eos = eos_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,   # (batch, seq_len)
        scores: torch.FloatTensor,     # (batch, vocab_size)
    ) -> torch.FloatTensor:
        # tokens the model has generated for the label so far
        generated = input_ids[0, self.prompt_length:].tolist()
        valid = self.trie.get_valid_next_tokens(generated)

        # if the generated prefix is already a complete label, allow EOS
        if self.trie.is_complete(generated):
            valid.add(self.eos)

        # set all invalid token logits to -inf
        masked = torch.full_like(scores, float("-inf"))
        for tid in valid:
            masked[0, tid] = scores[0, tid]

        return masked


class MultiLabelTrieLogitsProcessor(LogitsProcessor):
    """
    Multi-label constrained decoding.

    Extends single-label decoding by:
      1. Allowing EOS *or* a separator token after every complete label.
         The model picks whichever has higher probability — it is not forced
         to keep emitting labels.  EOS is always the primary stop mechanism.
      2. If the model picks the separator, already-emitted labels are excluded
         at ROOT so the next pick cannot repeat a previously chosen label.
      3. Safety net only: if the model emits a separator even though no unseen
         labels remain, EOS is added so generation can terminate cleanly.

    Parameters
    ----------
    trie          : ConstrainedTrie built from your label set.
    prompt_length : Number of tokens in the prompt.
    eos_token_id  : Token ID for end-of-sequence.
    sep_token_ids : Token IDs for the separator string (e.g. tokenizer.encode(", ", add_special_tokens=False)).
    """

    def __init__(
        self,
        trie: ConstrainedTrie,
        prompt_length: int,
        eos_token_id: int,
        sep_token_ids: list[int],
    ) -> None:
        self.trie = trie
        self.prompt_length = prompt_length
        self.eos = eos_token_id
        self.sep = sep_token_ids

        # precompute: first_token -> set of all complete label tuples that start with it
        # used to determine when a first token is fully "exhausted"
        self._first_to_labels: dict[int, set[tuple[int, ...]]] = {}
        self._build_first_token_map()

    # ------------------------------------------------------------------
    # Build-time precomputation
    # ------------------------------------------------------------------

    def _build_first_token_map(self) -> None:
        def dfs(node: TrieNode, path: list[int]) -> None:
            if node.is_end and path:
                first = path[0]
                self._first_to_labels.setdefault(first, set()).add(tuple(path))
            for tid, child in node.children.items():
                dfs(child, path + [tid])

        dfs(self.trie.root, [])

    # ------------------------------------------------------------------
    # Per-step helpers
    # ------------------------------------------------------------------

    def _parse_state(
        self, generated: list[int]
    ) -> tuple[set[tuple[int, ...]], list[int]]:
        """
        Split `generated` on separator token boundaries.

        Returns
        -------
        completed_labels : set of tuples, one per fully-emitted label.
        current_prefix   : token list for the label currently being built
                           (everything after the last separator).
        """
        sep, sep_len = self.sep, len(self.sep)
        completed: set[tuple[int, ...]] = set()
        seg_start = 0
        i = 0
        while i <= len(generated) - sep_len:
            if generated[i : i + sep_len] == sep:
                seg = tuple(generated[seg_start:i])
                if seg:
                    completed.add(seg)
                seg_start = i + sep_len
                i = seg_start
            else:
                i += 1
        return completed, generated[seg_start:]

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_length:].tolist()
        completed_labels, current_prefix = self._parse_state(generated)

        # valid continuations for the label currently being built
        valid = self.trie.get_valid_next_tokens(current_prefix)

        if self.trie.is_complete(current_prefix):
            valid.add(self.eos)          # option A: stop here
            for tid in self.sep:         # option B: separator → pick next label
                valid.add(tid)

        # at ROOT (reached after a separator): remove first tokens of labels
        # that have already been emitted, so the model cannot repeat them.
        # Normal termination happens above — the model picks EOS at any end node.
        # This block only runs when the model chose to continue via separator.
        if not current_prefix and completed_labels:
            exhausted_first_tokens = {
                first_tok
                for first_tok, label_set in self._first_to_labels.items()
                if label_set.issubset(completed_labels)
            }
            valid -= exhausted_first_tokens

            # safety net: model emitted a separator with no unseen labels left.
            # add EOS so generation terminates rather than getting stuck.
            if not valid:
                valid.add(self.eos)

        masked = torch.full_like(scores, float("-inf"))
        for tid in valid:
            masked[0, tid] = scores[0, tid]

        return masked
