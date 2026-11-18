"""Rule baseline: an offensive-word lexicon matched with minimum edit distance (technique area B).

The lexicon is learned from the training split only (build_lexicon): a word enters the lexicon when it
occurs in at least `min_count` training tweets and at least `min_ratio` of those tweets are offensive.
A message is flagged when one of its words is within the allowed edit distance of a lexicon word.
"""
import re
from collections import Counter
from functools import lru_cache

from chatmod.normalize import asciify, normalize

_WORD_RE = re.compile(r"[a-z]+")
_PLACEHOLDERS = {"user", "url"}


def tokenize(text: str) -> list[str]:
    """normalize -> asciify -> lowercase ASCII word tokens (placeholders removed)."""
    return [w for w in _WORD_RE.findall(asciify(normalize(text))) if w not in _PLACEHOLDERS]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein minimum edit distance (insertion, deletion, substitution all cost 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def allowed_distance(word: str) -> int:
    """Edits tolerated for a lexicon word: short words must match exactly."""
    if len(word) < 5:
        return 0
    return 1 if len(word) < 8 else 2


def build_lexicon(texts, labels, min_count: int, min_ratio: float) -> set[str]:
    total, offensive = Counter(), Counter()
    for text, label in zip(texts, labels):
        for w in set(tokenize(text)):
            total[w] += 1
            offensive[w] += label
    return {w for w, n in total.items() if n >= min_count and offensive[w] / n >= min_ratio}


class LexiconMatcher:
    def __init__(self, lexicon: set[str], prefix_match: bool = False):
        """prefix_match: also compare lexicon words of >= 5 letters with the beginning of the token,
        so Turkish suffixes are tolerated ("aptal" matches "aptalsin", "aptallar")."""
        self.lexicon = set(lexicon)
        self.prefix_match = prefix_match
        self._best = lru_cache(maxsize=200_000)(self._best_match)

    def _best_match(self, token: str):
        if token in self.lexicon:
            return token, 0
        best = None
        for word in self.lexicon:
            limit = allowed_distance(word)
            candidates = [token]
            if self.prefix_match and len(word) >= 5 and len(token) > len(word):
                candidates.append(token[:len(word)])
            for cand in candidates:
                if abs(len(word) - len(cand)) > limit or (limit == 0 and cand != word):
                    continue
                d = edit_distance(cand, word)
                if d <= limit and (best is None or d < best[1]):
                    best = (word, d)
        return best

    def matches(self, text: str) -> list[tuple[str, str, int]]:
        found = []
        for token in tokenize(text):
            hit = self._best(token)
            if hit:
                found.append((token, hit[0], hit[1]))
        return found

    def predict(self, text: str) -> int:
        return int(bool(self.matches(text)))
