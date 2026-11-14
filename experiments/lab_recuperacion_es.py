"""Lab: la primera búsqueda de vstash en español y con consultas vagas.

Cuatro bancos, una tabla por banco y por configuración de búsqueda:

  siempre  720 notas inventadas en español (90 asuntos de 8 notas que se confunden) con la
           consulta que solo responde cada una y que comparte palabras con ella.
  vagas    las mismas notas con consultas de 3 a 8 palabras sin nombres ni palabras de la nota.
  miracl   MIRACL-es (mteb/MIRACLRetrievalHardNegatives): 648 preguntas reales con relevancia
           juzgada por personas sobre 146.750 pasajes de Wikipedia, indexados «título. texto».
  scifact  SciFact de BEIR en inglés con bge-small (línea base documentada NDCG@10 = 0,7263).
  nfcorpus NFCorpus de BEIR, segundo control en inglés (consultas muy cortas).

Las notas y MIRACL se leen del laboratorio de rotten (solo lectura); el modelo del índice es
el mismo que allí. Los vectores se guardan en experiments/data/lab_es/ para no recalcularlos,
y cada medida deja el resultado consulta a consulta para poder comparar emparejado después.

    python experiments/lab_recuperacion_es.py medir --bancos siempre vagas --configs defecto vec
    python experiments/lab_recuperacion_es.py tabla [--bancos ...] [--configs ...] [--md]
    python experiments/lab_recuperacion_es.py resumen --configs defecto 'rec+sw'
    python experiments/lab_recuperacion_es.py tiempos --bancos siempre --configs defecto a-fondo

Los intervalos son de remuestreo emparejado contra la configuración «defecto» (5.000
remuestreos; por asunto en las notas, que van en racimos de 8; por consulta en el resto).
Siempre índices temporales: nunca se abre ~/.vstash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from vstash.embed import embed_query, embed_texts, get_embedding_dim  # noqa: E402
from vstash.store import VstashStore  # noqa: E402

ROTTEN_LAB = Path(os.environ.get("LAB_ES_DATOS", "/Users/jaysonsteffens/code/rotten/memoria/lab"))
CACHE = AQUI / "data" / "lab_es"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Configuraciones                                                      #
# ------------------------------------------------------------------ #
# «buscar»: argumentos de VstashStore.search. «almacen»: argumentos del constructor (índice
# distinto). «plan»: nombre de una estrategia compuesta de PLANES.

CONFIGS: dict[str, dict] = {
    "defecto": {},
    "vec": {"buscar": {"retrieval_mode": "vec_only"}},
    "fts": {"buscar": {"retrieval_mode": "fts_only"}},
    "sin-adapt": {"buscar": {"adaptive_rrf": False}},
    "corte2": {"buscar": {"distance_cutoff": 2.0}},
    "vec-corte2": {"buscar": {"retrieval_mode": "vec_only", "distance_cutoff": 2.0}},
    "sin-adapt-corte2": {"buscar": {"adaptive_rrf": False, "distance_cutoff": 2.0}},
}


# --- Experimento 1: tokenizador FTS, raíces y palabras vacías ---------------------------
TOKENIZADORES = {
    "porter-u61": "porter unicode61 remove_diacritics 2",
    "u61": "unicode61 remove_diacritics 2",
    "trigram": "trigram remove_diacritics 1",
}
for _n, _t in TOKENIZADORES.items():
    CONFIGS[f"tok-{_n}"] = {"almacen": {"fts_tokenizer": _t}}
CONFIGS["u61+snowball"] = {"almacen": {"fts_tokenizer": TOKENIZADORES["u61"]}, "texto": "snowball"}
CONFIGS["vacias"] = {"texto": "vacias"}
CONFIGS["tok-porter-u61+vacias"] = {**CONFIGS["tok-porter-u61"], "texto": "vacias"}
CONFIGS["tok-u61+vacias"] = {**CONFIGS["tok-u61"], "texto": "vacias"}
CONFIGS["u61+snowball+vacias"] = {**CONFIGS["u61+snowball"], "texto": "snowball+vacias"}
for _n in [c for c in CONFIGS if c.startswith(("tok-", "u61+", "vacias"))]:
    for _sufijo, _kw in (("sin-adapt", {"adaptive_rrf": False}), ("fts", {"retrieval_mode": "fts_only"})):  # fmt: skip
        CONFIGS[f"{_n}/{_sufijo}"] = {**CONFIGS[_n], "buscar": _kw}

# Palabras vacías: las listas de Snowball (español e inglés), recortadas a lo gramatical.
VACIAS = set(
    """a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
    durante e el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos
    esta estaba estado estamos estan estar estas este esto estos estoy fue fueron fui ha
    habia han has hasta hay la las le les lo los mas me mi mis mucho muy nada ni no nos
    nosotros o os otra otras otro otros para pero poco por porque que quien quienes se sea
    ser si sido sin sobre sois somos son soy su sus tambien te tenia tiene tienen todo todos
    tu tus un una uno unos vosotros y ya yo cuales cuanto cuanta cuantos cuantas cuyo
    the of and to in is was were be been are for on with as by at from that this which or
    an it its not but has have had do does did what who whom when where why how than then
    there their they them these those can could should would will may might we our you your
    he she his her i my me into about over under between after before during also""".split()  # noqa: SIM905
)


def _sin_acentos(t: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _palabras(t: str) -> list[str]:
    import re

    return re.findall(r"\w+", t.lower())


def _raices(t: str) -> str:
    import snowballstemmer

    if "es" not in _raices.__dict__:
        _raices.__dict__["es"] = snowballstemmer.stemmer("spanish")
    return " ".join(_raices.__dict__["es"].stemWords(_palabras(t)))


def _quitar_vacias(t: str) -> str:
    quedan = [w for w in _palabras(t) if _sin_acentos(w) not in VACIAS]
    return " ".join(quedan) if quedan else t


# nombre -> (transformación del texto indexado o None, transformación de la consulta).
# Los vectores salen siempre del texto original: solo cambia lo que ve FTS5.
TEXTOS: dict[str, tuple[Callable | None, Callable]] = {
    "snowball": (_raices, _raices),
    "vacias": (None, _quitar_vacias),
    "snowball+vacias": (_raices, lambda t: _raices(_quitar_vacias(t))),
}


# --- Experimento 2: reglas de pesos de la fusión (prototipos fuera de vstash) -----------
def _regla(ausentes: str, cobertura: bool = False, vacias: bool = False) -> Callable:
    """La regla de vstash (sigmoide de la IDF media) cambiando qué se hace con los términos
    de la consulta que no están en el vocabulario del índice: «max» = como hoy, cuentan
    como rarísimos; «fuera» = no cuentan; «cero» = cuentan como IDF 0. Con «cobertura» el
    tramo variable del peso se multiplica por la fracción de términos presentes."""

    def pesos(almacen: VstashStore, consulta: str) -> tuple[float, float]:
        idf, n = almacen._build_idf_cache()
        palabras = [w for w in consulta.split() if len(w) > 1]
        if vacias:
            palabras = [w for w in palabras if _sin_acentos(w.lower()).strip("¿?¡!.,;:") not in VACIAS]  # fmt: skip
        terminos = almacen._stem_terms(palabras) if palabras else []
        if n < 2 or not terminos:
            return 0.6, 0.4
        presentes = [idf[t] for t in terminos if t in idf]
        n_aus = len(terminos) - len(presentes)
        if ausentes == "max":
            valores = presentes + [math.log(n)] * n_aus
        elif ausentes == "cero":
            valores = presentes + [0.0] * n_aus
        else:
            valores = presentes
        if not valores:
            return 0.9, 0.1
        media = sum(valores) / len(valores)
        senal = 1.0 / (1.0 + math.exp(-2.0 * (media - math.log(n) / 2)))
        if cobertura:
            senal *= len(presentes) / len(terminos)
        fts = 0.1 + senal * 0.5
        return round(1.0 - fts, 3), round(fts, 3)

    return pesos


PESOS: dict[str, Callable] = {
    "max": _regla("max"),  # control: debe dar lo mismo que «defecto»
    "fuera": _regla("fuera"),
    "cero": _regla("cero"),
    "fuera-cob": _regla("fuera", cobertura=True),
    "max-cob": _regla("max", cobertura=True),
    "fuera-vacias": _regla("fuera", vacias=True),
    "fuera-cob-vacias": _regla("fuera", cobertura=True, vacias=True),
}


def _tope(maximo: float, base: str = "max") -> Callable:
    """La regla base, sin dejar que las palabras pesen más que «maximo»."""

    def pesos(almacen: VstashStore, consulta: str) -> tuple[float, float]:
        fts = min(PESOS[base](almacen, consulta)[1], maximo)
        return round(1.0 - fts, 3), fts

    return pesos


for _m in (0.5, 0.45, 0.4, 0.3):
    PESOS[f"tope-{_m}"] = _tope(_m)
PESOS["tope-0.45-cob"] = _tope(0.45, "max-cob")


def fuerza_lexica(almacen: VstashStore, consulta: str) -> float:
    """BM25 del mejor acierto por palabras, en unidades de ln N: cuántas «palabras raras»
    de la consulta casa, más o menos, la nota que mejor casa."""
    n = almacen._build_idf_cache()[1]
    casar, _ = almacen._build_fts_match_query(consulta)
    try:
        fila = almacen._conn.execute(
            "SELECT rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT 1", [casar]
        ).fetchone()
    except Exception:
        fila = None
    return -float(fila[0]) / math.log(n) if fila and n > 1 else 0.0


def _fuerza(base: str, bajo: float, alto: float) -> Callable:
    """El peso de las palabras de la regla base, reducido hacia 0,1 cuando el mejor acierto
    por palabras es flojo (rampa lineal entre «bajo» y «alto»)."""

    def pesos(almacen: VstashStore, consulta: str) -> tuple[float, float]:
        fts = PESOS[base](almacen, consulta)[1]
        rampa = min(1.0, max(0.0, (fuerza_lexica(almacen, consulta) - bajo) / (alto - bajo)))
        fts = round(0.1 + (fts - 0.1) * rampa, 3)
        return round(1.0 - fts, 3), fts

    return pesos


PESOS["fuerza"] = _fuerza("max", 1.0, 2.0)
PESOS["fuerza-tope"] = _fuerza("tope-0.45", 1.0, 2.0)
PESOS["fuerza-1.25-1.75"] = _fuerza("max", 1.25, 1.75)
PESOS["fuerza-1-1.5"] = _fuerza("max", 1.0, 1.5)
PESOS["fuerza-1.5-2.5"] = _fuerza("max", 1.5, 2.5)
for _n in PESOS:
    CONFIGS[f"regla-{_n}"] = {"pesos": _n}
# --- Experimento 3: corte por distancia (constantes) sobre la regla con tope ------------
# Las mismas dos ideas ya dentro de vstash, como opciones de search (apagadas por defecto).
CONFIGS["compuerta"] = {"buscar": {"lexical_gate": True}}
CONFIGS["compuerta-corte2"] = {"buscar": {"lexical_gate": True, "distance_cutoff": 2.0}}
CONFIGS["compuerta-suelo"] = {"buscar": {"lexical_gate": True, "min_vec_keep": 50}}
CONFIGS["compuerta-suelo20"] = {"buscar": {"lexical_gate": True, "min_vec_keep": 20}}
CONFIGS["suelo"] = {"buscar": {"min_vec_keep": 50}}
CONFIGS["suelo20"] = {"buscar": {"min_vec_keep": 20}}
CONFIGS["vec-suelo20"] = {"buscar": {"retrieval_mode": "vec_only", "min_vec_keep": 20}}
# Combinaciones de la candidata (compuerta + corte 2,0) con las piezas del experimento 1.
_REC = {"lexical_gate": True, "distance_cutoff": 2.0}
CONFIGS["rec+vacias"] = {"texto": "vacias", "buscar": _REC}
CONFIGS["rec+porter-u61"] = {**CONFIGS["tok-porter-u61"], "buscar": _REC}
CONFIGS["rec+porter-u61+vacias"] = {**CONFIGS["tok-porter-u61+vacias"], "buscar": _REC}
CONFIGS["rec+snowball"] = {**CONFIGS["u61+snowball"], "buscar": _REC}
CONFIGS["rec+snowball+vacias"] = {**CONFIGS["u61+snowball+vacias"], "buscar": _REC}
# La versión de dentro de vstash de «vacias»: search(drop_stopwords=True).
CONFIGS["sw"] = {"buscar": {"drop_stopwords": True}}
CONFIGS["rec+sw"] = {"buscar": {**_REC, "drop_stopwords": True}}
CONFIGS["rec+sw+porter-u61"] = {
    **CONFIGS["tok-porter-u61"],
    "buscar": {**_REC, "drop_stopwords": True},
}
CONFIGS["rec+sw/sin-adapt"] = {"buscar": {**_REC, "drop_stopwords": True, "adaptive_rrf": False}}
CONFIGS["sw-corte2"] = {"buscar": {"distance_cutoff": 2.0, "drop_stopwords": True}}
CONFIGS["sw-corte2/sin-adapt"] = {
    "buscar": {"distance_cutoff": 2.0, "drop_stopwords": True, "adaptive_rrf": False}
}
for _c in (1.5, 2.0, 25.0):
    CONFIGS[f"fuerza-corte{_c:g}"] = {"pesos": "fuerza", "buscar": {"distance_cutoff": _c}}
for _c in (1.3225, 1.5, 1.75, 2.0, 3.0, 25.0):
    CONFIGS[f"tope45-corte{_c:g}"] = {"pesos": "tope-0.45", "buscar": {"distance_cutoff": _c}}
    CONFIGS[f"vec-corte{_c:g}"] = {"buscar": {"retrieval_mode": "vec_only", "distance_cutoff": _c}}
for _f in (0.1, 0.2, 0.3, 0.5):
    CONFIGS[f"fijo-{_f}"] = {"buscar": {"vec_weight": 1 - _f, "fts_weight": _f}}


def _ids_union(almacen, vector, consulta, top_k, **kw):
    """Híbrido y solo vectores (corte 2,0), intercalados y sin repetir: dos búsquedas."""
    a = almacen.search(vector, consulta, top_k=top_k, **kw)
    b = almacen.search(
        vector, consulta, top_k=top_k, retrieval_mode="vec_only", distance_cutoff=2.0
    )
    vistos: dict[str, None] = {}
    for par in zip(a, b):
        for r in par:
            vistos.setdefault(r.path)
    for r in [*a, *b]:
        vistos.setdefault(r.path)
    return list(vistos)


PLANES: dict[str, Callable] = {"union": _ids_union}

# --- Experimento 4: búsqueda «a fondo» (unión de híbrido y solo vectores) ---------------
# «top_k» en la configuración cambia cuántos se piden; la unión pide top_k a cada pata.
CONFIGS["union"] = {"plan": "union"}
CONFIGS["union-compuerta"] = {"plan": "union", "buscar": {"lexical_gate": True}}
CONFIGS["defecto-100"] = {"top_k": 100}
CONFIGS["corte2-100"] = {"top_k": 100, "buscar": {"distance_cutoff": 2.0}}
CONFIGS["compuerta-corte2-100"] = {
    "top_k": 100,
    "buscar": {"lexical_gate": True, "distance_cutoff": 2.0},
}
CONFIGS["defecto-40"] = {"top_k": 40}
CONFIGS["compuerta-corte2-40"] = {
    "top_k": 40,
    "buscar": {"lexical_gate": True, "distance_cutoff": 2.0},
}
CONFIGS["vec-corte2-100"] = {
    "top_k": 100,
    "buscar": {"retrieval_mode": "vec_only", "distance_cutoff": 2.0},
}


# ------------------------------------------------------------------ #
# Bancos                                                               #
# ------------------------------------------------------------------ #


class Banco:
    nombre: str
    modelo: str
    top_k: int
    docs: list[str]
    consultas: list[str]
    buenas: list[dict[int, int]]  # por consulta: índice de documento -> ganancia
    grupos: np.ndarray  # racimo de cada consulta para el remuestreo
    persistir = False


def _notas(nombre: str, archivo: str) -> Banco:
    filas = json.loads((ROTTEN_LAB / archivo).read_text())
    b = Banco()
    b.nombre, b.modelo, b.top_k = nombre, MINILM, 50
    b.docs = [f["nota"] for f in filas]
    b.consultas = [f["consulta"] for f in filas]
    b.buenas = [{i: 1} for i in range(len(filas))]
    asuntos: dict[str, int] = {}
    b.grupos = np.array([asuntos.setdefault(f["asunto"], len(asuntos)) for f in filas])
    return b


def _miracl() -> Banco:
    aqui = ROTTEN_LAB / "comparar" / "miracl"
    corpus = json.loads((aqui / "corpus.json").read_text())
    consultas = json.loads((aqui / "queries.json").read_text())
    donde = {d["id"]: i for i, d in enumerate(corpus)}
    cual = {q["id"]: i for i, q in enumerate(consultas)}
    b = Banco()
    b.nombre, b.modelo, b.top_k, b.persistir = "miracl", MINILM, 20, True
    b.docs = [f"{d['title']}. {d['text']}" for d in corpus]
    b.consultas = [q["text"] for q in consultas]
    b.buenas = [{} for _ in consultas]
    for r in json.loads((aqui / "qrels.json").read_text()):
        if r["score"] > 0:
            b.buenas[cual[r["query-id"]]][donde[r["corpus-id"]]] = int(r["score"])
    b.grupos = np.arange(len(consultas))
    return b


def _beir(nombre: str) -> Banco:
    from experiments.beir_benchmark import download_beir, load_beir

    anterior = os.getcwd()
    os.chdir(AQUI.parent)  # beir_benchmark usa una ruta relativa para su caché
    try:
        corpus, consultas, qrels = load_beir(download_beir(nombre))
    finally:
        os.chdir(anterior)
    ids = list(corpus)
    donde = {d: i for i, d in enumerate(ids)}
    b = Banco()
    b.nombre, b.modelo, b.top_k = nombre, BGE, 10
    b.docs = [(corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in ids]
    b.consultas = [consultas[q] for q in qrels]
    b.buenas = [{donde[d]: s for d, s in qrels[q].items() if d in donde} for q in qrels]
    b.grupos = np.arange(len(b.consultas))
    return b


BANCOS: dict[str, Callable[[], Banco]] = {
    "siempre": lambda: _notas("siempre", "notas.json"),
    "vagas": lambda: _notas("vagas", "notas_vagas.json"),
    "miracl": _miracl,
    "scifact": lambda: _beir("scifact"),
    # Segundo control en inglés: consultas de dos o tres palabras, donde la compuerta
    # léxica actúa más que en SciFact.
    "nfcorpus": lambda: _beir("nfcorpus"),
}


# ------------------------------------------------------------------ #
# Vectores e índice                                                    #
# ------------------------------------------------------------------ #


def _vectores(b: Banco, que: str) -> np.ndarray:
    """Vectores de los documentos o de las consultas, guardados en disco."""
    CACHE.mkdir(parents=True, exist_ok=True)
    textos = b.docs if que == "docs" else b.consultas
    huella = hashlib.sha1(("\x00".join(textos) + b.modelo).encode()).hexdigest()[:10]
    ruta = CACHE / f"vec_{b.nombre}_{que}_{huella}.npy"
    if ruta.exists():
        return np.load(ruta, mmap_mode="r")
    empieza = time.perf_counter()
    if que == "docs":
        partes = []
        for i in range(0, len(textos), 1024):
            partes.append(np.asarray(embed_texts(textos[i : i + 1024], b.modelo), dtype=np.float32))
            if len(textos) > 20000 and (i // 1024) % 10 == 0:
                print(f"    vectores {i}/{len(textos)} · {time.perf_counter() - empieza:.0f} s")
        v = np.concatenate(partes)
    else:
        v = np.asarray([embed_query(t, b.modelo) for t in textos], dtype=np.float32)
    np.save(ruta, v)
    print(
        f"  {b.nombre}: {len(textos)} {que} vectorizados en {time.perf_counter() - empieza:.0f} s"
    )
    return v


_carpetas: list[tempfile.TemporaryDirectory] = []


def indice(b: Banco, almacen_kw: dict, texto: str | None = None) -> VstashStore:
    """Índice de prueba: temporal, o guardado en la caché del lab para el corpus grande."""
    de_doc = TEXTOS[texto][0] if texto else None
    clave = hashlib.sha1(
        json.dumps([almacen_kw, texto if de_doc else None], sort_keys=True).encode()
    ).hexdigest()[:8]
    if b.persistir:
        ruta = CACHE / f"idx_{b.nombre}_{clave}.db"
        if ruta.exists():
            return VstashStore(str(ruta), embedding_dim=get_embedding_dim(b.modelo), **almacen_kw)
    else:
        carpeta = tempfile.TemporaryDirectory()
        _carpetas.append(carpeta)
        ruta = Path(carpeta.name) / "prueba.db"
    v = _vectores(b, "docs")
    parcial = ruta.with_suffix(".parcial.db") if b.persistir else ruta
    for resto in parcial.parent.glob(parcial.name + "*") if b.persistir else []:
        resto.unlink()
    almacen = VstashStore(str(parcial), embedding_dim=v.shape[1], **almacen_kw)
    empieza = time.perf_counter()
    for i in range(0, len(b.docs), 5000):
        almacen.add_documents_batch(
            [
                {
                    "path": f"/doc/{j}",
                    "title": "",
                    "chunks": [de_doc(b.docs[j]) if de_doc else b.docs[j]],
                    "embeddings": [v[j].tolist()],
                    "source_type": "text",
                }
                for j in range(i, min(i + 5000, len(b.docs)))
            ]
        )
    if b.persistir:
        almacen.close()
        parcial.rename(ruta)
        almacen = VstashStore(str(ruta), embedding_dim=v.shape[1], **almacen_kw)
        print(f"  {b.nombre}: índice {clave} montado en {time.perf_counter() - empieza:.0f} s")
    return almacen


# ------------------------------------------------------------------ #
# Medir                                                                #
# ------------------------------------------------------------------ #


def buscar(almacen: VstashStore, cfg: dict, vector: list[float], consulta: str, k: int):
    kw = cfg.get("buscar", {})
    if "texto" in cfg:
        consulta = TEXTOS[cfg["texto"]][1](consulta)
    if "pesos" in cfg:
        vw, fw = PESOS[cfg["pesos"]](almacen, consulta)
        kw = {**kw, "vec_weight": vw, "fts_weight": fw}
    if "plan" in cfg:
        rutas = PLANES[cfg["plan"]](almacen, vector, consulta, k, **kw)
    else:
        rutas = [r.path for r in almacen.search(vector, consulta, top_k=k, **kw)]
    return [int(p.rsplit("/", 1)[1]) for p in rutas]


def _ndcg10(dados: list[int], buenas: dict[int, int]) -> float:
    dcg = sum(buenas.get(d, 0) / math.log2(i + 2) for i, d in enumerate(dados[:10]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(buenas.values(), reverse=True)[:10]))  # fmt: skip
    return dcg / ideal if ideal else 0.0


def _salida(banco: str, config: str) -> Path:
    return CACHE / "resultados" / f"{banco}__{config.replace('/', '_')}.npz"


def medir(bancos: list[str], configs: list[str]) -> None:
    for nombre in bancos:
        b = BANCOS[nombre]()
        vq = _vectores(b, "consultas")
        por_indice: dict[str, list[str]] = {}
        for c in configs:
            t = CONFIGS[c].get("texto")
            clave = [CONFIGS[c].get("almacen", {}), t if t and TEXTOS[t][0] else None]
            por_indice.setdefault(json.dumps(clave, sort_keys=True), []).append(c)
        for kw_json, lista in por_indice.items():
            almacen = indice(b, *json.loads(kw_json))
            almacen.search(vq[0].tolist(), b.consultas[0], top_k=b.top_k)  # calentar
            for c in lista:
                puesto, n, ndcg, ms = [], [], [], []
                for i, consulta in enumerate(b.consultas):
                    vector = vq[i].tolist()
                    empieza = time.perf_counter()
                    dados = buscar(
                        almacen, CONFIGS[c], vector, consulta, CONFIGS[c].get("top_k", b.top_k)
                    )
                    ms.append((time.perf_counter() - empieza) * 1e3)
                    puesto.append(next((k + 1 for k, d in enumerate(dados) if d in b.buenas[i]), 0))
                    n.append(len(dados))
                    ndcg.append(_ndcg10(dados, b.buenas[i]))
                ruta = _salida(nombre, c)
                ruta.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    ruta,
                    puesto=np.array(puesto),
                    n=np.array(n),
                    ndcg=np.array(ndcg),
                    ms=np.array(ms),
                    grupos=b.grupos,
                )
                print(linea(nombre, c), flush=True)
            almacen.close()


# ------------------------------------------------------------------ #
# Tabla                                                                #
# ------------------------------------------------------------------ #

# Columnas por banco: (título, función sobre el resultado guardado -> valor por consulta)
_en = lambda k: lambda r: ((r["puesto"] > 0) & (r["puesto"] <= k)).astype(float)  # noqa: E731
COLUMNAS: dict[str, list[tuple[str, Callable]]] = {
    "siempre": [
        ("primera", _en(1)),
        ("en 5", _en(5)),
        ("en 20", _en(20)),
        ("en 50", _en(50)),
        ("entra", _en(10**6)),
    ],
    "vagas": [
        ("primera", _en(1)),
        ("en 5", _en(5)),
        ("en 20", _en(20)),
        ("en 50", _en(50)),
        ("entra", _en(10**6)),
    ],
    "miracl": [
        ("primera", _en(1)),
        ("en 3", _en(3)),
        ("en 20", _en(20)),
        ("NDCG@10", lambda r: r["ndcg"]),
        ("entra", _en(10**6)),
    ],
    "scifact": [("NDCG@10", lambda r: r["ndcg"]), ("primera", _en(1)), ("en 10", _en(10))],
    "nfcorpus": [("NDCG@10", lambda r: r["ndcg"]), ("primera", _en(1)), ("en 10", _en(10))],
}


def _intervalo(dif: np.ndarray, grupos: np.ndarray, veces: int = 5000) -> tuple[float, float]:
    """Remuestreo emparejado por racimos: se sortean grupos enteros con reposición."""
    _, inv = np.unique(grupos, return_inverse=True)
    suma = np.bincount(inv, weights=dif)
    cuenta = np.bincount(inv).astype(float)
    rng = np.random.default_rng(1)
    elegidos = rng.integers(0, len(suma), size=(veces, len(suma)))
    medias = suma[elegidos].sum(1) / cuenta[elegidos].sum(1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def _celdas(banco: str, config: str) -> list[str] | None:
    ruta = _salida(banco, config)
    if not ruta.exists():
        return None
    r = np.load(ruta)
    base = np.load(_salida(banco, "defecto")) if _salida(banco, "defecto").exists() else None
    celdas = []
    for _, f in COLUMNAS[banco]:
        v = f(r)
        texto = f"{v.mean() * 100:.1f}"
        if base is not None and config != "defecto":
            dif = v - f(base)
            lo, hi = _intervalo(dif, r["grupos"])
            marca = "" if lo <= 0 <= hi else " *"
            texto += f" ({dif.mean() * 100:+.1f}; {lo * 100:+.1f} a {hi * 100:+.1f}{marca})"
        celdas.append(texto)
    celdas.append(f"{r['n'].mean():.1f}")
    celdas.append(f"{np.median(r['ms']):.1f}")
    return celdas


def linea(banco: str, config: str) -> str:
    celdas = _celdas(banco, config)
    titulos = [t for t, _ in COLUMNAS[banco]] + ["devuelve", "ms"]
    return f"  [{banco}] {config:24} " + " · ".join(f"{t} {c}" for t, c in zip(titulos, celdas or []))  # fmt: skip


def tabla(bancos: list[str], configs: list[str], md: bool) -> None:
    for banco in bancos:
        titulos = [t for t, _ in COLUMNAS[banco]] + ["devuelve", "ms"]
        print(f"\n### {banco}\n" if md else f"\n== {banco}")
        if md:
            print("| Configuración | " + " | ".join(titulos) + " |")
            print("|---|" + "---|" * len(titulos))
        for c in configs:
            celdas = _celdas(banco, c)
            if celdas is None:
                continue
            print(f"| {c} | " + " | ".join(celdas) + " |" if md else linea(banco, c))


# Medidas principales de cada banco, para la tabla de una fila por configuración.
RESUMEN: list[tuple[str, str, Callable]] = [
    ("siempre", "siempre 1ª", _en(1)),
    ("siempre", "siempre en 20", _en(20)),
    ("vagas", "vagas 1ª", _en(1)),
    ("vagas", "vagas en 20", _en(20)),
    ("vagas", "vagas en 50", _en(50)),
    ("miracl", "MIRACL 1ª", _en(1)),
    ("miracl", "MIRACL en 20", _en(20)),
    ("scifact", "SciFact NDCG@10", lambda r: r["ndcg"]),
    ("nfcorpus", "NFCorpus NDCG@10", lambda r: r["ndcg"]),
]


def resumen(configs: list[str]) -> None:
    """Una fila por configuración: valor, diferencia con «defecto» y * si el intervalo del
    95 % de esa diferencia no contiene el cero. «—» = no medido."""
    print("| Configuración | " + " | ".join(t for _, t, _ in RESUMEN) + " |")
    print("|---|" + "---|" * len(RESUMEN))
    for c in configs:
        celdas = []
        for banco, _, f in RESUMEN:
            if not _salida(banco, c).exists():
                celdas.append("—")
                continue
            r, base = np.load(_salida(banco, c)), np.load(_salida(banco, "defecto"))
            v = f(r)
            texto = f"{v.mean() * 100:.1f}"
            if c != "defecto":
                dif = v - f(base)
                lo, hi = _intervalo(dif, r["grupos"])
                texto += f" ({dif.mean() * 100:+.1f}{'' if lo <= 0 <= hi else ' *'})"
            celdas.append(texto.replace(".", ","))
        if any(x != "—" for x in celdas):
            print(f"| `{c}` | " + " | ".join(celdas) + " |")


def tiempos(bancos: list[str], configs: list[str], vueltas: int = 5) -> None:
    """Milisegundos por búsqueda (sin vectorizar la consulta): el menor de varias vueltas."""
    for nombre in bancos:
        b = BANCOS[nombre]()
        vq = _vectores(b, "consultas")
        for c in configs:
            almacen = indice(b, CONFIGS[c].get("almacen", {}), CONFIGS[c].get("texto"))
            cuantas = min(len(b.consultas), 200)
            mejores = []
            for _ in range(vueltas):
                empieza = time.perf_counter()
                for i in range(cuantas):
                    k = CONFIGS[c].get("top_k", b.top_k)
                    buscar(almacen, CONFIGS[c], vq[i].tolist(), b.consultas[i], k)
                mejores.append((time.perf_counter() - empieza) / cuantas * 1e3)
            print(f"  [{nombre}] {c:24} {min(mejores):.2f} ms por búsqueda (menor de {vueltas})")
            almacen.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("orden", choices=["medir", "tabla", "resumen", "tiempos"])
    p.add_argument("--bancos", nargs="*", default=["siempre", "vagas"])
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--md", action="store_true")
    a = p.parse_args()
    import fnmatch

    # Cada nombre puede ser un patrón: --configs 'tok-*' 'vacias*'
    cfgs = [c for patron in (a.configs or ["*"]) for c in CONFIGS if fnmatch.fnmatch(c, patron)]
    if a.orden == "medir":
        medir(a.bancos, cfgs)
    elif a.orden == "resumen":
        resumen(cfgs)
    elif a.orden == "tiempos":
        tiempos(a.bancos, cfgs)
    else:
        tabla(a.bancos, cfgs, a.md)
