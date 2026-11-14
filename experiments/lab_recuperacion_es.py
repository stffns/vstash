"""Lab: la primera búsqueda de vstash en español y con consultas vagas.

Cuatro bancos, una tabla por banco y por configuración de búsqueda:

  siempre  720 notas inventadas en español (90 asuntos de 8 notas que se confunden) con la
           consulta que solo responde cada una y que comparte palabras con ella.
  vagas    las mismas notas con consultas de 3 a 8 palabras sin nombres ni palabras de la nota.
  miracl   MIRACL-es (mteb/MIRACLRetrievalHardNegatives): 648 preguntas reales con relevancia
           juzgada por personas sobre 146.750 pasajes de Wikipedia, indexados «título. texto».
  scifact  SciFact de BEIR en inglés con bge-small (línea base documentada NDCG@10 = 0,7263).

Las notas y MIRACL se leen del laboratorio de rotten (solo lectura); el modelo del índice es
el mismo que allí. Los vectores se guardan en experiments/data/lab_es/ para no recalcularlos,
y cada medida deja el resultado consulta a consulta para poder comparar emparejado después.

    python experiments/lab_recuperacion_es.py medir --bancos siempre vagas --configs defecto vec
    python experiments/lab_recuperacion_es.py tabla [--bancos ...] [--configs ...] [--md]
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


def _scifact() -> Banco:
    from experiments.beir_benchmark import download_beir, load_beir

    anterior = os.getcwd()
    os.chdir(AQUI.parent)  # beir_benchmark usa una ruta relativa para su caché
    try:
        corpus, consultas, qrels = load_beir(download_beir("scifact"))
    finally:
        os.chdir(anterior)
    ids = list(corpus)
    donde = {d: i for i, d in enumerate(ids)}
    b = Banco()
    b.nombre, b.modelo, b.top_k = "scifact", BGE, 10
    b.docs = [(corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in ids]
    b.consultas = [consultas[q] for q in qrels]
    b.buenas = [{donde[d]: s for d, s in qrels[q].items() if d in donde} for q in qrels]
    b.grupos = np.arange(len(b.consultas))
    return b


BANCOS: dict[str, Callable[[], Banco]] = {
    "siempre": lambda: _notas("siempre", "notas.json"),
    "vagas": lambda: _notas("vagas", "notas_vagas.json"),
    "miracl": _miracl,
    "scifact": _scifact,
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


def indice(b: Banco, almacen_kw: dict) -> VstashStore:
    """Índice de prueba: temporal, o guardado en la caché del lab para el corpus grande."""
    clave = hashlib.sha1(json.dumps(almacen_kw, sort_keys=True).encode()).hexdigest()[:8]
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
                    "chunks": [b.docs[j]],
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
    return CACHE / "resultados" / f"{banco}__{config}.npz"


def medir(bancos: list[str], configs: list[str]) -> None:
    for nombre in bancos:
        b = BANCOS[nombre]()
        vq = _vectores(b, "consultas")
        por_indice: dict[str, list[str]] = {}
        for c in configs:
            por_indice.setdefault(json.dumps(CONFIGS[c].get("almacen", {}), sort_keys=True), []).append(c)  # fmt: skip
        for kw_json, lista in por_indice.items():
            almacen = indice(b, json.loads(kw_json))
            almacen.search(vq[0].tolist(), b.consultas[0], top_k=b.top_k)  # calentar
            for c in lista:
                puesto, n, ndcg, ms = [], [], [], []
                for i, consulta in enumerate(b.consultas):
                    vector = vq[i].tolist()
                    empieza = time.perf_counter()
                    dados = buscar(almacen, CONFIGS[c], vector, consulta, b.top_k)
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
    "siempre": [("primera", _en(1)), ("en 5", _en(5)), ("en 20", _en(20)), ("en 50", _en(50))],
    "vagas": [("primera", _en(1)), ("en 5", _en(5)), ("en 20", _en(20)), ("en 50", _en(50))],
    "miracl": [
        ("primera", _en(1)),
        ("en 3", _en(3)),
        ("en 20", _en(20)),
        ("NDCG@10", lambda r: r["ndcg"]),
    ],
    "scifact": [("NDCG@10", lambda r: r["ndcg"]), ("primera", _en(1)), ("en 10", _en(10))],
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


def tiempos(bancos: list[str], configs: list[str], vueltas: int = 5) -> None:
    """Milisegundos por búsqueda (sin vectorizar la consulta): el menor de varias vueltas."""
    for nombre in bancos:
        b = BANCOS[nombre]()
        vq = _vectores(b, "consultas")
        for c in configs:
            almacen = indice(b, CONFIGS[c].get("almacen", {}))
            cuantas = min(len(b.consultas), 200)
            mejores = []
            for _ in range(vueltas):
                empieza = time.perf_counter()
                for i in range(cuantas):
                    buscar(almacen, CONFIGS[c], vq[i].tolist(), b.consultas[i], b.top_k)
                mejores.append((time.perf_counter() - empieza) / cuantas * 1e3)
            print(f"  [{nombre}] {c:24} {min(mejores):.2f} ms por búsqueda (menor de {vueltas})")
            almacen.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("orden", choices=["medir", "tabla", "tiempos"])
    p.add_argument("--bancos", nargs="*", default=["siempre", "vagas"])
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--md", action="store_true")
    a = p.parse_args()
    cfgs = a.configs or list(CONFIGS)
    if a.orden == "medir":
        medir(a.bancos, cfgs)
    elif a.orden == "tiempos":
        tiempos(a.bancos, cfgs)
    else:
        tabla(a.bancos, cfgs, a.md)
