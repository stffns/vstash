"""Function words dropped from the keyword side of a query (opt-in).

Used only by ``search(drop_stopwords=True)``: the words are removed from the
FTS5 MATCH expression and from the adaptive-RRF IDF average. Nothing is
removed from the index, so the option needs no reindex and can be turned on
and off per call.

Spanish and English only, and deliberately short: articles, prepositions,
conjunctions, pronouns, question words and the auxiliary forms of
ser / estar / haber / be / have / do. Words that double as content words are
left out on purpose ("estado", "era", "nada", "will", "may", "can", "who"),
and accents are significant: "te" is dropped, "té" is not; "el" and "él" are
both listed because both are grammatical. Matching is case-insensitive.
"""

from __future__ import annotations

_SPANISH = """
a al ante bajo con contra de del desde durante en entre hacia hasta para por segun según
sin sobre tras
el la los las lo un una unos unas
y e o u ni que pero sino porque pues aunque si como cuando donde mientras
yo tu tú el él ella ello ellos ellas nosotros nosotras vosotros vosotras usted ustedes
me te se nos os le les mi mis tus su sus
este esta estos estas ese esa esos esas eso esto aquel aquella aquellos aquellas
qué quien quién quienes quiénes cual cuál cuales cuáles cuándo dónde cómo cuanto cuánto
cuanta cuánta cuantos cuántos cuantas cuántas cuyo cuya cuyos cuyas
es son soy eres somos sois fue fueron ser sido siendo
está están estoy estamos estaba estaban estar
ha han he has hay había habían haber
no sí también tambien más mas muy ya
"""

_ENGLISH = """
a an the of in on at by for from to with without into onto over under between about
and or but nor if than then that this these those there
i you he she it we they me him her us them my your his its our their
what which whom whose when where why how
is are was were be been being am
has have had do does did
not also
"""

STOPWORDS: frozenset[str] = frozenset((_SPANISH + _ENGLISH).split())

_PUNCTUATION = "¿?¡!.,;:()[]{}\"'«»“”‘’…"


def is_stopword(word: str) -> bool:
    """True for a listed function word, ignoring case and surrounding punctuation."""
    return word.strip(_PUNCTUATION).lower() in STOPWORDS


def drop_stopwords(words: list[str]) -> list[str]:
    """Remove stopwords; a query made only of stopwords is returned unchanged."""
    kept = [w for w in words if not is_stopword(w)]
    return kept or words
