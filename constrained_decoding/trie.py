"""
trie.py
-------
Prefix-tree (trie) over tokenized label sequences.

Build once from your taxonomy, then query at every decode step to get the
set of valid next tokens. Rebuild whenever the label set changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class ConstrainedTrie:
    """
    Trie that stores every valid label as a path of token IDs.

    Usage
    -----
    trie = ConstrainedTrie()
    for label in labels:
        token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        trie.insert(token_ids)
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def insert(self, token_ids: list[int]) -> None:
        """Insert one label (as a list of token IDs) into the trie."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_valid_next_tokens(self, prefix: list[int]) -> set[int]:
        """
        Return the set of token IDs that are valid continuations of `prefix`.

        Returns an empty set if `prefix` is not a valid path in the trie
        (this should never happen when constraints are working correctly).
        """
        node = self._walk(prefix)
        if node is None:
            return set()
        return set(node.children.keys())

    def is_complete(self, prefix: list[int]) -> bool:
        """Return True if `prefix` is a complete label (ends at an end node)."""
        node = self._walk(prefix)
        return node is not None and node.is_end

    # ------------------------------------------------------------------
    # Introspection (used by MultiLabelTrieLogitsProcessor)
    # ------------------------------------------------------------------

    def get_all_label_sequences(self) -> list[tuple[int, ...]]:
        """Return every root-to-end path as a tuple of token IDs."""
        results: list[tuple[int, ...]] = []

        def dfs(node: TrieNode, path: list[int]) -> None:
            if node.is_end and path:
                results.append(tuple(path))
            for tid, child in node.children.items():
                dfs(child, path + [tid])

        dfs(self.root, [])
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def verify(self, labels: list[str], tokenizer) -> bool:
        """
        Check that every label round-trips correctly through the trie.
        Call this once after build time — it is the correctness proof.

        Returns True if the trie exactly encodes the given labels.
        Raises ValueError with details if anything is wrong.
        """
        expected = set()
        for label in labels:
            tids = tuple(tokenizer.encode(" " + label, add_special_tokens=False))
            decoded = tokenizer.decode(list(tids)).strip()
            if decoded != label:
                raise ValueError(
                    f"Tokenization round-trip failed for '{label}': "
                    f"got '{decoded}' — check leading-space tokenization"
                )
            expected.add(tids)

        actual = set(self.get_all_label_sequences())
        if actual != expected:
            missing = expected - actual
            extra   = actual - expected
            raise ValueError(
                f"Trie mismatch.\n"
                f"  Missing paths : {missing}\n"
                f"  Unexpected paths: {extra}"
            )
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _walk(self, prefix: list[int]) -> TrieNode | None:
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return None
            node = node.children[tid]
        return node
