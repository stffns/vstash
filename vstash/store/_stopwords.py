"""Function words dropped from the keyword side of a query (opt-in).

Used only by ``search(drop_stopwords=True)``: the words are removed from the
FTS5 MATCH expression and from the adaptive-RRF IDF average. Nothing is
removed from the index, so the option needs no reindex and can be turned on
and off per call. Spanish and English only, grammatical words only -- the
lists are deliberately short so that no content word is ever dropped.
Entries are lowercase with accents folded; compare via ``is_stopword``.
"""

from __future__ import annotations

import unicodedata

_SPANISH = """
a al algo algunas algunos ante antes como con contra cual cuales cuando cuanta cuantas
cuanto cuantos cuyo de del desde donde durante e el ella ellas ellos en entre era erais
eran eras eres es esa esas ese eso esos esta estaba estado estamos estan estar estas este
esto estos estoy fue fueron fui ha habia han has hasta hay la las le les lo los mas me mi
mis mucho muy nada ni no nos nosotros o os otra otras otro otros para pero poco por porque
que quien quienes se sea ser si sido sin sobre sois somos son soy su sus tambien te tenia
tiene tienen todo todos tu tus un una uno unos vosotros y ya yo
"""

_ENGLISH = """
a about after also an and are as at be been before between but by can could did do does
during for from had has have he her his how i in into is it its may me might my not of on
or our over she should than that the their them then there these they this those to under
was we were what when where which who whom why will with would you your
"""

STOPWORDS: frozenset[str] = frozenset((_SPANISH + _ENGLISH).split())

_PUNCTUATION = "¿?¡!.,;:()[]{}\"'«»“”‘’…"


def _fold(word: str) -> str:
    """Lowercase, strip surrounding punctuation and remove diacritics."""
    decomposed = unicodedata.normalize("NFD", word.strip(_PUNCTUATION).lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def is_stopword(word: str) -> bool:
    return _fold(word) in STOPWORDS


def drop_stopwords(words: list[str]) -> list[str]:
    """Remove stopwords; a query made only of stopwords is returned unchanged."""
    kept = [w for w in words if not is_stopword(w)]
    return kept or words
