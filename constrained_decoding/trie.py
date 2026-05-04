"""
trie.py — prefix tree over tokenized label sequences.

Build once from your taxonomy, query at every decode step to get the set
of valid next tokens. Rebuild whenever the label set changes.
"""


class TrieNode:
    def __init__(self):
        self.children = {}   # token_id → TrieNode
        self.is_end   = False


class ConstrainedTrie:
    """
    Stores every valid label as a path of token IDs.

    Usage
    -----
    trie = ConstrainedTrie()
    for label in labels:
        token_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        trie.insert(token_ids)
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids):
        """Add one label (list of token IDs) to the trie."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True

    def get_valid_next_tokens(self, prefix):
        """Which tokens can the model emit next, given what it has emitted so far."""
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return set()
            node = node.children[tid]
        return set(node.children.keys())

    def is_complete(self, prefix):
        """True if prefix exactly spells out one of the inserted labels."""
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end

    def all_labels(self):
        """Return every stored label as a tuple of token IDs."""
        results = []

        def walk(node, path):
            if node.is_end:
                results.append(tuple(path))
            for tid, child in node.children.items():
                walk(child, path + [tid])

        walk(self.root, [])
        return results

    def verify(self, labels, tokenizer):
        """
        Check every label round-trips correctly through the trie.
        Call once after build — this is the build-time correctness proof.
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

        actual = set(self.all_labels())
        if actual != expected:
            raise ValueError(
                f"Trie mismatch.\n"
                f"  Missing  : {expected - actual}\n"
                f"  Unexpected: {actual - expected}"
            )
        return True
