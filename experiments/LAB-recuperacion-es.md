# Lab: la primera búsqueda de vstash en español y con consultas vagas

14 de noviembre de 2026 · rama `pelops/lab-recuperacion-es-ef71b8` · vstash 0.39.0 (develop, `bb6d5bc`)

## Configuración recomendada (provisional, 14-nov 20:20; quedan experimentos por cerrar)

**No pierde en ninguno de los cinco bancos** y es la que debe usar la prueba unificada:

```python
# rama pelops/lab-recuperacion-es-ef71b8 (las tres opciones son nuevas y están apagadas por defecto)
almacen.search(
    vector_de_la_consulta, texto_de_la_consulta, top_k=K,
    lexical_gate=True,      # baja el peso de las palabras cuando la evidencia léxica es floja
    distance_cutoff=2.0,    # corte por distancia más flojo que el 1,3225 por defecto
    drop_stopwords=True,    # «de», «la», «the»… fuera de la parte de palabras
)
```

Si hay que usar vstash 0.39.0 tal cual, sin esta rama: `search(..., adaptive_rrf=False,
distance_cutoff=2.0)`. Recoge casi toda la ganancia en las vagas y en MIRACL, pero pierde
algo en NFCorpus (alguna buena entre 10: −3,1 puntos, intervalo sin el cero) y no mejora la
primera en las consultas de siempre.

| Banco (medida) | Por defecto | Recomendada | Sin tocar código |
|---|---|---|---|
| siempre: primera · entre 20 | 72,8 · 95,0 | **74,9** (+2,1; +0,1 a +4,2 *) · 96,2 (+1,2; 0,0 a +2,5) | 71,9 (−0,8; −2,1 a +0,4) · 96,4 (+1,4 *) |
| vagas: primera · entre 20 · entre 50 | 5,6 · 36,5 · 59,3 | 9,2 (+3,6 *) · **55,3** (+18,8; +14,9 a +22,6 *) · 74,0 (+14,7 *) | 5,3 · 51,5 (+15,0 *) · 71,1 (+11,8 *) |
| MIRACL-es: primera · entre 20 | 38,1 · 81,9 | 39,8 (+1,7; −0,8 a +4,2) · **87,8** (+5,9; +3,4 a +8,5 *) | 40,6 (+2,5; −0,9 a +5,7) · 89,2 (+7,3 *) |
| SciFact: NDCG@10 | 72,5 | 73,1 (+0,6; −0,3 a +1,6) | 73,2 (+0,7; −0,9 a +2,3) |
| NFCorpus: NDCG@10 | 35,9 | 36,1 (+0,2; −0,3 a +0,7) | 35,4 (−0,5; −1,3 a +0,2) |
| ms por búsqueda (notas · SciFact · MIRACL) | 2,8 · 8,1 · 205 | 1,2 · 5,0 · 88 | 2,9 · 8,2 · 193 |

Avisos: (1) `lexical_gate`, `drop_stopwords` y `min_vec_keep` solo existen en
`VstashStore.search` de esta rama. (2) Con consultas vagas la buena sigue sin ser la primera
más que el 9 %: lo que mejora es que entra en la lista; el orden fino sigue siendo trabajo del
reordenador. (3) No hace falta reindexar nada: las tres opciones son de consulta.

<!-- RECOMENDACION -->

## Cómo se midió

Banco: `experiments/lab_recuperacion_es.py`. Cinco conjuntos, siempre con índices temporales:

| Banco | Qué es | Modelo del índice | Se piden | Medida principal |
|---|---|---|---|---|
| siempre | 720 notas inventadas en español (90 asuntos de 8 notas que se confunden); cada consulta comparte nombres y palabras con su nota | MiniLM multilingüe | 50 | buena la primera, entre 20 |
| vagas | las mismas notas con consultas de 3 a 8 palabras sin nombres ni palabras de la nota (extremo artificial) | MiniLM multilingüe | 50 | entre 20 |
| miracl | MIRACL-es (`mteb/MIRACLRetrievalHardNegatives`): 648 preguntas reales, relevancia juzgada por personas, 146.750 pasajes | MiniLM multilingüe | 20 | primera, entre 20 |
| scifact | SciFact de BEIR, inglés, 300 consultas | bge-small-en | 10 | NDCG@10 |
| nfcorpus | NFCorpus de BEIR, inglés, 323 consultas muy cortas (se añadió como segundo control) | bge-small-en | 10 | NDCG@10 |

- Un trozo por documento, vectores calculados una vez y guardados; lo único que cambia entre
  filas es la llamada a `search` (o el tokenizador del índice).
- Entre paréntesis va la diferencia con la configuración por defecto y su intervalo del 95 %
  por remuestreo **emparejado** (5.000 remuestreos; en las notas se sortean asuntos enteros,
  porque van en racimos de 8; en el resto, consultas). Un asterisco marca los intervalos que
  no contienen el cero.
- «devuelve» es cuántos resultados da de media. Se vigila porque aflojar filtros cambia
  recuerdo por ruido.
- **Criterio de «no pierde»**: en la medida principal de cada banco, el intervalo de la
  diferencia no queda entero por debajo de cero. Las reglas se ajustaron mirando solo las
  notas; MIRACL y los dos de inglés se usaron de puerta, no para ajustar. El criterio se
  escribió después de ver las notas y SciFact pero antes de tener MIRACL y NFCorpus.
- Tiempos: milisegundos de `search` sin contar vectorizar la consulta; el menor de cinco
  vueltas, con la máquina sin otro trabajo.

<!-- PARTIDA -->

<!-- EXPERIMENTOS -->

## 5. La etiqueta de `miss_analysis` — arreglado

**Qué pasaba.** `_mmr_dedup(ranked, top_k)` hace dos cosas: deduplica dentro del documento y
recorta a `top_k`. El trazado miraba si el trozo seguía en la lista después de esa llamada y,
si no, lo apuntaba como «eliminado por la deduplicación MMR». Con un trozo por documento la
deduplicación no puede eliminar nada: lo que pasaba es que el trozo había quedado por debajo
del puesto pedido. La rama `top_k_cutoff` con `passed=False` era inalcanzable.

**Cambio** (`vstash/store/_search.py`). Solo cuando hay trazado, la deduplicación se ejecuta
sin recortar y se recorta después. La selección es voraz y estable en su prefijo, así que los
`top_k` primeros son exactamente los mismos. Con eso se distinguen tres casos:

| Antes de MMR | Después de MMR (sin recortar) | Veredicto |
|---|---|---|
| por debajo de `top_k` | sigue por debajo | `top_k_cutoff` (MMR pasa) |
| dentro de `top_k` | empujado fuera por la penalización de un hermano | `mmr_dedup` |
| estaba | ya no está (deduplicación dura) | `mmr_dedup` |

**Tests** (`tests/test_miss_analysis.py::TestBelowTopKIsNotMmr`): con un trozo por documento
el veredicto es `top_k_cutoff` y la sugerencia no habla de MMR (falla sin el arreglo); el
trazado devuelve los mismos resultados que `search`; un hermano casi idéntico sigue dando
`mmr_dedup`.

**Coste.** Ninguno en la búsqueda normal. En el trazado, la selección recorre toda la lista de
candidatos en vez de `top_k`: es diagnóstico, no camino caliente.

**Consecuencia para el diagnóstico de rotten.** Los 224 fallos «por debajo del puesto 20 tras
la fusión» estaban bien contados (rotten ya los leía así); lo que cambia es que ahora vstash
los llama por su nombre.

## 6. La orden `vstash` que falla con `click_type` — es del entorno instalado

**Causa.** typer 0.29 quitó el argumento `click_type` de `typer.Option`. El `_retrain.py`
antiguo declaraba tres opciones con `click_type=click.Choice([...])`; como los decoradores se
evalúan al importar `vstash.cli`, el `TypeError` tumbaba **todas** las órdenes, no solo
`retrain`.

**Dónde está.** En el repositorio ya está arreglado: `develop` lo corrigió en `095caad`
(#521) cambiando esas opciones a `Enum`. La orden instalada falla porque la herramienta de uv
es una instalación **editable** que apunta a `~/code/vstash`, y esa copia está en la rama
`memoria-confiable`, que no contiene `095caad`; su entorno tiene typer 0.29.0.

| | typer | ¿acepta `click_type`? | `_retrain.py` |
|---|---|---|---|
| herramienta instalada (`~/.local/share/uv/tools/vstash`) | 0.29.0 | no | el de `memoria-confiable`: usa `click_type` → falla |
| este repositorio (develop) | 0.29.1 | no | usa `Enum` → funciona |

**Arreglo para Jay** (no lo he hecho: es su instalación y su rama): traer develop a
`memoria-confiable` (`git merge develop` o `git rebase develop` en `~/code/vstash`). No hace
falta reinstalar: al ser editable, la orden vuelve a funcionar en cuanto el archivo cambie.

**Lo que sí añadí.** El arreglo de #521 no traía test, y el pin `typer>=0.12.0` no tiene techo,
así que puede volver a pasar con otra versión. `tests/test_cli_typer_compat.py` comprueba que
todas las órdenes (35) construyen su `--help` con el typer instalado, que ningún módulo de la
CLI pasa `click_type` cuando typer no lo acepta, y que las tres opciones de elección siguen
rechazando valores desconocidos.

<!-- CIERRE -->
