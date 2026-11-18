"""Regex-based normalization of noisy Turkish chat text (technique area B).

normalize(): used by every model.
    1. Turkish-aware lowercasing (I -> ı, İ -> i)
    2. (the dataset already replaces mentions/links with @USER / URL; they become "@user" / "url")
    3. leetspeak decoding: a digit or symbol is replaced only when it sits between two letters
       ("s4l4k" -> "salak"), so real numbers ("3000") and hashtags ("#1mayıs") are kept
    4. letters repeated 3+ times are collapsed to one ("aptaaaal" -> "aptal");
       legitimate Turkish double letters ("saat") are untouched
    5. whitespace squeezing
asciify(): maps Turkish letters to ASCII (ç->c, ğ->g, ı->i, ö->o, ş->s, ü->u). Used by the classical
    models and the lexicon matcher so that "gerizekalı" and "gerizekali" become the same word.
"""
import re

LETTER = "a-zçğıöşüâîû"
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
_LEET_RE = re.compile(rf"(?<=[{LETTER}])[{re.escape(''.join(LEET))}](?=[{LETTER}])")
_REPEAT_RE = re.compile(rf"([{LETTER}])\1{{2,}}")
_ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _decode_leet(token: str) -> str:
    if token.startswith("#"):
        return token
    return _LEET_RE.sub(lambda m: LEET[m.group(0)], token)


def normalize(text: str) -> str:
    text = turkish_lower(text)
    tokens = [_decode_leet(t) for t in text.split()]
    text = " ".join(tokens)
    text = _REPEAT_RE.sub(r"\1", text)
    return " ".join(text.split())


def normalization_steps(text: str) -> list[str]:
    """Names of the normalize() rules that change this text (shown in the app; reveals no words)."""
    steps = []
    if "I" in text or "İ" in text:
        steps.append("Turkish casing")
    lowered = turkish_lower(text)
    decoded = " ".join(_decode_leet(t) for t in lowered.split())
    if decoded != " ".join(lowered.split()):
        steps.append("leetspeak")
    if _REPEAT_RE.search(decoded):
        steps.append("repeated letters")
    return steps


def asciify(text: str) -> str:
    return text.translate(_ASCII)


def classical_preprocess(text: str) -> str:
    """Input of the classical and word2vec models: normalized + asciified text."""
    return asciify(normalize(text))
