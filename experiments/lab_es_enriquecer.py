"""Lab: enriquecer las notas al guardarlas, con un modelo local, para la búsqueda por palabras.

Dos pasadas independientes sobre las notas del laboratorio de rotten (solo lectura). El modelo
solo trabaja al guardar; al consultar no hay ningún modelo.

  campos    numind/NuExtract3-mlx-4bits saca de cada nota personas, organizaciones y
            proyectos, fechas, cantidades y un asunto de una línea (experimento 8).
  terminos  un modelo de chat pequeño (mlx-community/gemma-4-e2b-it-4bit) escribe palabras y
            expresiones con las que alguien buscaría la nota sin recordar sus nombres ni sus
            cifras: términos que la nota no contiene (experimento 7).

    uv run --offline --no-project --with mlx-vlm python experiments/lab_es_enriquecer.py campos viejas
    uv run --offline --no-project --with mlx-lm  python experiments/lab_es_enriquecer.py terminos viejas

«viejas» son las 720 notas de ajuste; «final», las 840 del examen. La salida va a
experiments/data/lab_es/enriquecido/ y se puede reanudar. Un solo modelo cargado cada vez.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nada de red: los modelos ya están en el equipo

AQUI = Path(__file__).resolve().parent
ROTTEN_LAB = Path(os.environ.get("LAB_ES_DATOS", "/Users/jaysonsteffens/code/rotten/memoria/lab"))
SALIDA = AQUI / "data" / "lab_es" / "enriquecido"
CONJUNTOS = {"viejas": ROTTEN_LAB / "notas.json", "final": ROTTEN_LAB / "final" / "notas.json"}

NUEXTRACT = "numind/NuExtract3-mlx-4bits"
FICHA = {
    "personas": ["verbatim-string"],
    "organizaciones_y_proyectos": ["verbatim-string"],
    "fechas": ["verbatim-string"],
    "cantidades": ["verbatim-string"],
    "asunto": "string",
}
INSTRUCCIONES = (
    "El texto es una nota breve de trabajo en español. «personas»: nombres de persona tal"
    " como aparecen. «organizaciones_y_proyectos»: nombres propios de empresas, equipos,"
    " productos, sistemas o proyectos. «fechas»: fechas y plazos tal como aparecen."
    " «cantidades»: cifras con su unidad (dinero, porcentajes, duraciones, recuentos)."
    " «asunto»: de qué trata la nota, en una frase corta con palabras corrientes."
)

CHAT = "mlx-community/gemma-4-e2b-it-4bit"
PETICION = (
    "Esta es una nota guardada en una memoria personal de trabajo:\n\n«{nota}»\n\n"
    "Dentro de unos meses alguien la buscará sin recordar los nombres propios ni las cifras,"
    " con una frase vaga. Escribe 12 palabras o expresiones cortas (de una a tres palabras)"
    " que esa persona podría usar: el tema, el tipo de decisión o de problema, sinónimos y"
    " palabras más generales que las de la nota. No repitas nombres propios ni cifras de la"
    " nota. Responde solo con las expresiones, separadas por comas, en español."
)


def _notas(conjunto: str) -> list[str]:
    return [f["nota"] for f in json.loads(CONJUNTOS[conjunto].read_text())]


def _reanudar(ruta: Path) -> list:
    return json.loads(ruta.read_text()) if ruta.exists() else []


def _guardar(
    ruta: Path, hechas: list, empieza: float, nuevas: int, total: int, muestra: str
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(hechas, ensure_ascii=False))
    por_nota = (time.perf_counter() - empieza) / max(1, nuevas)
    print(f"{len(hechas)}/{total} · {por_nota:.2f} s por nota · {muestra[:150]!r}", flush=True)


def campos(conjunto: str) -> None:
    from mlx_vlm import generate, load

    notas = _notas(conjunto)
    ruta = SALIDA / f"campos_{conjunto}.json"
    hechas = _reanudar(ruta)
    modelo, procesador = load(NUEXTRACT)

    def encargo(texto: str) -> str:
        return (
            "<|im_start|>user\n【task】structured\n"
            f"【template_start】{json.dumps(FICHA, ensure_ascii=False, indent=4)}【template_end】\n"
            f"【instructions_start】{INSTRUCCIONES}【instructions_end】\n"
            f"【document_start】\n{texto}\n【document_end】<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    generate(modelo, procesador, encargo("Hola."), max_tokens=40, verbose=False)
    empieza, nuevas = time.perf_counter(), 0
    for i in range(len(hechas), len(notas)):
        dado = generate(modelo, procesador, encargo(notas[i]), max_tokens=220, verbose=False)
        texto = dado if isinstance(dado, str) else dado.text
        try:
            ficha = json.loads(texto[texto.index("{") : texto.rindex("}") + 1])
        except ValueError:
            ficha = {}
        hechas.append(ficha)
        nuevas += 1
        if nuevas % 40 == 0 or i == len(notas) - 1:
            _guardar(
                ruta, hechas, empieza, nuevas, len(notas), json.dumps(ficha, ensure_ascii=False)
            )


def terminos(conjunto: str) -> None:
    from mlx_lm import generate, load

    notas = _notas(conjunto)
    ruta = SALIDA / f"terminos_{conjunto}.json"
    hechas = _reanudar(ruta)
    modelo, tokenizador = load(CHAT)
    empieza, nuevas = time.perf_counter(), 0
    for i in range(len(hechas), len(notas)):
        mensaje = [{"role": "user", "content": PETICION.format(nota=notas[i])}]
        entrada = tokenizador.apply_chat_template(
            mensaje, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        texto = generate(modelo, tokenizador, prompt=entrada, max_tokens=120, verbose=False)
        hechas.append(texto.strip())
        nuevas += 1
        if nuevas % 40 == 0 or i == len(notas) - 1:
            _guardar(ruta, hechas, empieza, nuevas, len(notas), texto.strip())


if __name__ == "__main__":
    {"campos": campos, "terminos": terminos}[sys.argv[1]](sys.argv[2])
