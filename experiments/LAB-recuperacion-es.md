# Lab: la primera búsqueda de vstash en español y con consultas vagas

14 de noviembre de 2026 · rama `pelops/lab-recuperacion-es-ef71b8` · vstash 0.39.0 (develop, `bb6d5bc`)

## Configuración recomendada (actualizada 14-nov 20:55: fusión que no se pisa)

**Primera etapa para un reordenador** (lo que importa es que la buena entre entre las 50):

```python
# rama pelops/lab-recuperacion-es-ef71b8; lexical_gate es nueva y está apagada por defecto
almacen.search(
    vector_de_la_consulta, texto_de_la_consulta, top_k=50,
    adaptive_rrf=False,     # pesos fijos 0,6 vectores / 0,4 palabras
    distance_cutoff=2.0,    # sin corte por distancia a efectos prácticos
    lexical_gate=True,      # las palabras dejan de votar si no tienen ninguna evidencia
)
# por el SDK: Memory.search(consulta, top_k=50, vec_weight=0.6, fts_weight=0.4,
#                           distance_cutoff=2.0, lexical_gate=True)
```

**Búsqueda sola, sin reordenador** (lo que importa es el orden; no pierde en ningún banco):
la misma llamada sin `adaptive_rrf=False`.

Las dos se llaman `R50` y `R1` en el banco. `drop_stopwords` ya **no** va en la
recomendación (ver «El híbrido no se pisa»). Con vstash 0.39.0 sin esta rama:
`adaptive_rrf=False, distance_cutoff=2.0`, que es `R50` sin la compuerta (pierde el primer
puesto de las consultas vagas: 5,3 % frente a 11,9 %).

<!-- RESUMEN-FINAL -->

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

## Línea de partida, reproducida

Todas las cifras del encargo salen iguales con el banco nuevo (vstash 0.39.0 de la copia de
trabajo, comprobado con `vstash.__file__`), y MIRACL sale igual que en rotten (38,1 % ·
59,0 % en 3 · 81,9 % entre 20):

<!-- T: defecto vec fts sin-adapt corte2 vec-corte2 sin-adapt-corte2 -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `defecto` | 72,8 | 95,0 | 5,6 | 36,5 | 59,3 | 38,1 | 81,9 | 72,5 | 35,9 |
| `vec` | 55,6 (-17,2 *) | 89,2 (-5,8 *) | 16,0 (+10,4 *) | 52,2 (+15,7 *) | 62,5 (+3,2 *) | 32,9 (-5,2 *) | 69,0 (-13,0 *) | 69,6 (-2,9 *) | 30,7 (-5,2 *) |
| `fts` | 70,6 (-2,2) | 91,2 (-3,8 *) | 2,2 (-3,3 *) | 9,2 (-27,4 *) | 15,7 (-43,6 *) | — | — | 68,3 (-4,2 *) | 31,4 (-4,5 *) |
| `sin-adapt` | 71,8 (-1,0) | 95,0 (+0,0) | 7,1 (+1,5 *) | 51,7 (+15,1 *) | 65,0 (+5,7 *) | 40,7 (+2,6) | 82,6 (+0,6 *) | 72,4 (-0,1) | 35,2 (-0,7 *) |
| `corte2` | 73,2 (+0,4) | 95,6 (+0,6) | 4,2 (-1,4 *) | 34,9 (-1,7) | 60,7 (+1,4) | 37,8 (-0,3) | 84,6 (+2,6 *) | 73,3 (+0,8 *) | 36,2 (+0,3) |
| `vec-corte2` | 55,6 (-17,2 *) | 91,2 (-3,8 *) | 16,0 (+10,4 *) | 56,8 (+20,3 *) | 74,4 (+15,1 *) | 32,9 (-5,2 *) | 81,5 (-0,5) | 71,9 (-0,6) | 33,6 (-2,3 *) |
| `sin-adapt-corte2` | 71,9 (-0,8) | 96,4 (+1,4 *) | 5,3 (-0,3) | 51,5 (+15,0 *) | 71,1 (+11,8 *) | 40,6 (+2,5) | 89,2 (+7,3 *) | 73,2 (+0,7) | 35,4 (-0,5) |
<!-- /T -->

- SciFact da NDCG@10 = 0,7250 y no 0,7263. No es el banco: el guion original
  (`experiments/beir_benchmark.py --datasets scifact --no-chroma`) da 0,7251 en este mismo
  entorno, y `experiments/results/beir_benchmark.json` tiene apuntado 0,7286 de otra tanda.
  Son diferencias de entorno (versión de fastembed/onnxruntime) de una décima; todas las
  comparaciones de este informe son contra el 0,7250 medido aquí.
- El recuento de fallos de rotten también se reproduce, ya con la etiqueta arreglada
  (experimento 5): de 461 fallos con consultas vagas pidiendo 20, 193 se caen en el corte por
  distancia, 224 quedan por debajo del puesto 20 y 44 no aparecen; en 174 la buena estaba
  entre las 20 primeras por vectores.
- Lo que ya enseña esta tabla, antes de tocar nada: **quitar los pesos adaptativos también
  mejora MIRACL** (+2,6 en primera, de 0,0 a +5,2; NDCG@10 +1,4, de +0,8 a +2,1), no solo las
  consultas vagas. Y aflojar el corte mejora SciFact (+0,8, de +0,3 a +1,4) y MIRACL
  (entre 20 +2,6). Los dos valores por defecto están costando recuerdo fuera de las notas
  inventadas.

## 1. Tokenizador del índice de palabras, raíces en español y palabras vacías

**Qué hace hoy.** `tokenize='porter ascii'`: raíces del inglés, y todo carácter que no sea
ASCII cuenta como letra y no se pasa a minúscula. Comprobado: «¿Quién» se indexa como el
término único `¿quién`, «REUNIÓN» como `reuniÓn` y «Ángel» como `Ángel`. Una pregunta que
empieza por «¿» pierde su primera palabra, y una mayúscula acentuada no casa con su
minúscula. Además `_build_fts_match_query` mete en la consulta todas las palabras de más de
una letra, vacías incluidas.

**Hipótesis.** (a) Un tokenizador que pliegue acentos y mayúsculas arregla esos casos sin
tocar el inglés. (b) Raíces del español mejoran la parte de palabras. (c) Quitar palabras
vacías de la consulta limpia la parte de palabras.

**Cambios.** `VstashStore(fts_tokenizer=...)` con una lista cerrada de cuatro valores (va
dentro del DDL, así que no es texto libre). Raíces: Snowball en Python aplicado en el banco
al texto indexado y a la consulta (prototipo; ver «coste»). Palabras vacías:
`search(drop_stopwords=True)`, lista corta de español e inglés: artículos, preposiciones,
conjunciones, pronombres, interrogativos y formas auxiliares de ser/estar/haber. Se dejaron
fuera a propósito las que también son palabras de contenido («estado», «era», «nada»,
«will», «may», «who») y los acentos cuentan («te» se quita, «té» no). Una primera lista más
ancha, que plegaba acentos, daba las mismas cifras (±0,3) en los cuatro bancos ligeros.

### 1a. Tokenizador

<!-- T: tok-porter-u61 tok-u61 tok-trigram fts tok-porter-u61/fts tok-u61/fts tok-trigram/fts -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `tok-porter-u61` | 72,1 (-0,7) | 94,6 (-0,4) | 5,6 (+0,0) | 36,8 (+0,3) | 60,0 (+0,7 *) | 39,0 (+0,9) | 83,3 (+1,4 *) | 72,5 (+0,0) | 35,8 (-0,1) |
| `tok-u61` | 72,4 (-0,4) | 95,3 (+0,3) | 5,3 (-0,3) | 35,1 (-1,4) | 58,8 (-0,6) | — | — | 72,2 (-0,3) | 35,1 (-0,8 *) |
| `tok-trigram` | 67,9 (-4,9 *) | 92,8 (-2,2 *) | 10,6 (+5,0 *) | 53,9 (+17,4 *) | 65,4 (+6,1 *) | — | — | 71,1 (-1,4 *) | 34,2 (-1,7 *) |
| `fts` | 70,6 (-2,2) | 91,2 (-3,8 *) | 2,2 (-3,3 *) | 9,2 (-27,4 *) | 15,7 (-43,6 *) | — | — | 68,3 (-4,2 *) | 31,4 (-4,5 *) |
| `tok-porter-u61/fts` | 68,6 (-4,2 *) | 91,0 (-4,0 *) | 2,5 (-3,1 *) | 9,9 (-26,7 *) | 16,1 (-43,2 *) | — | — | 68,4 (-4,1 *) | — |
| `tok-u61/fts` | 68,8 (-4,0 *) | 91,0 (-4,0 *) | 2,2 (-3,3 *) | 8,9 (-27,6 *) | 15,4 (-43,9 *) | — | — | 65,5 (-7,0 *) | — |
| `tok-trigram/fts` | 62,8 (-10,0 *) | 86,8 (-8,2 *) | 3,1 (-2,5 *) | 9,0 (-27,5 *) | 16,0 (-43,3 *) | — | — | 60,9 (-11,6 *) | — |
<!-- /T -->

(`…/fts` es buscar solo por palabras: aísla el efecto del tokenizador.)

- **`porter unicode61 remove_diacritics 2`: positivo pequeño, y seguro para el inglés.**
  SciFact da exactamente lo mismo que hoy (ni una consulta cambia) y NFCorpus −0,1 (de −0,4
  a 0,0). En MIRACL, que es español real con acentos y preguntas con «¿», «entre 20» sube
  +1,4 (de +0,3 a +2,5) y NDCG@10 +0,9 (de +0,4 a +1,5). **En las notas inventadas no se
  nota** (primera −0,7, de −2,1 a +0,7): sus consultas son largas y perder la primera palabra
  pesa poco. Es el único tokenizador que podría ser valor por defecto algún día.
- **`unicode61` sin raíces: negativo.** Pierde las raíces del inglés: SciFact solo por
  palabras baja de 68,3 a 65,5 y NFCorpus híbrido −0,8 (intervalo sin el cero). En español
  no gana nada a cambio.
- **Trigramas: negativo.** Solo por palabras pierde 8 puntos en las consultas de siempre y
  7 en SciFact. La subida que muestra en las vagas con el híbrido es un artefacto: el
  vocabulario pasa a ser de trigramas, la IDF media se desploma y los pesos adaptativos dan
  casi todo a los vectores; es lo mismo que consigue la compuerta del experimento 2 sin
  romper lo demás.

### 1b. Raíces en español (Snowball)

<!-- T: u61+snowball u61+snowball/fts rec+snowball rec+snowball+vacias -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `u61+snowball` | 73,3 (+0,6) | 95,4 (+0,4) | 8,2 (+2,6 *) | 40,8 (+4,3 *) | 61,5 (+2,2 *) | — | — | 72,1 (-0,4) | 35,2 (-0,7 *) |
| `u61+snowball/fts` | 74,3 (+1,5) | 94,0 (-1,0) | 4,3 (-1,2) | 14,6 (-21,9 *) | 22,9 (-36,4 *) | — | — | 67,5 (-5,0 *) | — |
| `rec+snowball` | 73,8 (+1,0) | 96,1 (+1,1) | 10,0 (+4,4 *) | 57,5 (+21,0 *) | 74,4 (+15,1 *) | — | — | 72,4 (-0,1) | 35,7 (-0,2) |
| `rec+snowball+vacias` | 77,4 (+4,6 *) | 96,1 (+1,1) | 10,0 (+4,4 *) | 56,1 (+19,6 *) | 73,8 (+14,4 *) | — | — | 71,5 (-1,0) | 35,6 (-0,3) |
<!-- /T -->

<!-- SNOWBALL -->

### 1c. Palabras vacías fuera de la consulta

<!-- T: sw vacias/fts compuerta-corte2 rec+sw -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `sw` | 74,4 (+1,7) | 95,0 (+0,0) | 3,8 (-1,8 *) | 35,1 (-1,4) | 59,6 (+0,3) | 40,6 (+2,5 *) | 84,6 (+2,6 *) | 72,7 (+0,2) | 35,9 (+0,0) |
| `vacias/fts` | 69,4 (-3,3) | 91,1 (-3,9 *) | 1,8 (-3,8 *) | 6,1 (-30,4 *) | 7,1 (-52,2 *) | — | — | 68,1 (-4,4 *) | — |
| `compuerta-corte2` | 73,9 (+1,1) | 96,1 (+1,1) | 9,0 (+3,5 *) | 56,4 (+19,9 *) | 74,2 (+14,9 *) | 38,0 (-0,2) | 84,4 (+2,5 *) | 73,1 (+0,6) | 36,3 (+0,4) |
| `rec+sw` | 74,9 (+2,1 *) | 96,2 (+1,2) | 9,2 (+3,6 *) | 55,3 (+18,8 *) | 74,0 (+14,7 *) | 39,8 (+1,7) | 87,8 (+5,9 *) | 73,1 (+0,6) | 36,1 (+0,2) |
<!-- /T -->

- **Sola, con los pesos de hoy: mixto.** Mejora MIRACL (primera +2,5, de +0,6 a +4,3) y no
  toca el inglés, pero **empeora las vagas** (entre 5: −7,4). El motivo está en el
  experimento 2: sin palabras vacías la IDF media de la consulta sube, la regla adaptativa da
  más peso a las palabras, y con consultas vagas eso es justo lo que no hay que hacer.
- **Con la compuerta delante: positivo.** `rec+sw` frente a `compuerta-corte2`: siempre
  primera 73,9 → 74,9; MIRACL primera 38,0 → 39,8 y entre 20 84,4 → 87,8; vagas igual
  (56,4 → 55,3 entre 20, dentro del ruido); inglés igual.
- **Regalo:** la búsqueda por palabras deja de recorrer las listas enormes de «de», «la»,
  «the»; el tiempo de `search` baja a menos de la mitad (ver costes).
- Solo por palabras no ayuda (`vacias/fts`): la mejora viene de cómo cambia los pesos y de
  no meter ruido en la fusión, no de que BM25 ordene mejor.

**Convivencia con índices ya creados y con el inglés.**
- El tokenizador solo se aplica al crear el índice (`CREATE … IF NOT EXISTS`). Para que no
  haya desajustes, el almacén **lee del propio índice** qué tokenizador tiene y usa ese para
  calcular las raíces de la consulta (antes estaba escrito a mano en dos sitios). Pedir otro
  distinto sobre una base existente no rompe nada: avisa por el registro y sigue con el del
  índice.
- Para cambiarlo: `almacen.rebuild_fts("porter unicode61 remove_diacritics 2")`. Borra y
  rehace solo el índice de palabras a partir de la tabla `chunks`, en una transacción, sin
  tocar los vectores. No se midió el tiempo en una base grande; es una pasada por el texto.
- Para el inglés `porter unicode61` es indistinguible de `porter ascii` en los dos bancos.
- Las palabras vacías no tocan el índice: se puede activar y desactivar por llamada.

**Coste.** Tokenizador: cero en consulta. Palabras vacías: negativo (más rápido). Snowball:
ver 1b.

## 2. Pesos adaptativos de la fusión

**Qué hacen hoy.** `_compute_adaptive_rrf_params` calcula la IDF media de los términos de la
consulta y la pasa por una sigmoide centrada en ln(N)/2: palabras raras → hasta 0,6 de peso
para las palabras; palabras corrientes → 0,1. Un término que **no está en el vocabulario**
cuenta como rarísimo (IDF máxima).

**Por qué hunden las consultas vagas.** Tres cosas medidas:

| | siempre | vagas | SciFact | NFCorpus | MIRACL |
|---|---|---|---|---|---|
| Peso medio que reciben las palabras | 0,49 | 0,53 | 0,16 | 0,35 | 0,11 (mediana) |
| Consultas con más peso para palabras que para vectores | 52 % | 76 % | 1 % | 35 % | ~0 % |
| Términos de la consulta que no existen en el índice | 24 % | 31 % | 2 % | 7 % | 0,2 % |

1. En un índice pequeño casi todo es «raro»: el 76 % de las consultas vagas recibe más peso
   para las palabras que para los vectores, empujado además por ese 31 % de términos que no
   existen (no pueden casar con nada y aun así suben el peso de las palabras).
2. Con más de 0,5 para las palabras, **una nota que solo casa por palabras adelanta a una
   que solo casa por vectores en el mismo puesto**. Con 0,56/0,44, la buena en el puesto 6
   por vectores (0,44/65 = 0,0068) queda detrás de las 22 primeras notas por palabras. En el
   tramo de peso > 0,5 las vagas tienen la buena entre 20 el 35,0 % con la regla adaptativa
   y el 54,4 % con pesos fijos 0,6/0,4; en los demás tramos da igual.
3. En las consultas de siempre y en SciFact la regla **no aporta nada medible** (siempre:
   −1,0 en primera sin ella, de −2,1 a +0,1; SciFact −0,1). Donde sí aporta es en NFCorpus
   (−0,7 sin ella, de −1,3 a −0,1): consultas de dos o tres palabras raras.

Y un cuarto dato que no se buscaba: en MIRACL la regla falla **hacia el otro lado**. Con
146.750 pasajes el umbral ln(N)/2 es 5,95, las palabras vacías de la pregunta («es», «de»,
«la») tiran de la media hacia abajo y las palabras se quedan con 0,11 de peso; con pesos
fijos 0,6/0,4 MIRACL gana +2,6 en primera.

**Reglas probadas** (prototipos en el banco, pasando los pesos a mano; `regla-max` reproduce
la de vstash al decimal, como control):

<!-- T: sin-adapt fijo-0.1 fijo-0.3 fijo-0.5 regla-fuera regla-cero regla-max-cob regla-tope-0.45 regla-tope-0.4 compuerta -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `sin-adapt` | 71,8 (-1,0) | 95,0 (+0,0) | 7,1 (+1,5 *) | 51,7 (+15,1 *) | 65,0 (+5,7 *) | 40,7 (+2,6) | 82,6 (+0,6 *) | 72,4 (-0,1) | 35,2 (-0,7 *) |
| `fijo-0.1` | 65,0 (-7,8 *) | 92,6 (-2,4 *) | 11,8 (+6,2 *) | 53,1 (+16,5 *) | 65,8 (+6,5 *) | — | — | 72,3 (-0,2) | 34,4 (-1,5 *) |
| `fijo-0.3` | 71,0 (-1,8 *) | 94,9 (-0,1) | 8,3 (+2,8 *) | 53,3 (+16,8 *) | 65,4 (+6,1 *) | — | — | 73,0 (+0,5) | 35,2 (-0,7 *) |
| `fijo-0.5` | 72,4 (-0,4) | 95,8 (+0,8) | 7,4 (+1,8 *) | 47,4 (+10,8 *) | 62,4 (+3,1 *) | — | — | 72,4 (-0,1) | 35,7 (-0,2) |
| `regla-fuera` | 72,2 (-0,6) | 94,6 (-0,4) | 6,4 (+0,8) | 45,7 (+9,2 *) | 63,1 (+3,8 *) | — | — | 72,2 (-0,3) | 35,9 (-0,0) |
| `regla-cero` | 67,9 (-4,9 *) | 93,6 (-1,4 *) | 9,0 (+3,5 *) | 51,7 (+15,1 *) | 64,9 (+5,6 *) | — | — | 72,2 (-0,3) | 35,9 (+0,0) |
| `regla-max-cob` | 71,9 (-0,8) | 95,1 (+0,1) | 7,2 (+1,7 *) | 50,1 (+13,6 *) | 64,3 (+5,0 *) | — | — | 72,5 (-0,0) | 35,9 (+0,0) |
| `regla-tope-0.45` | 72,1 (-0,7) | 95,3 (+0,3) | 7,1 (+1,5 *) | 50,6 (+14,0 *) | 63,5 (+4,2 *) | — | — | 72,5 (+0,0) | 35,3 (-0,6 *) |
| `regla-tope-0.4` | 71,7 (-1,1) | 95,0 (+0,0) | 7,1 (+1,5 *) | 51,5 (+15,0 *) | 64,9 (+5,6 *) | — | — | 72,5 (+0,0) | 35,3 (-0,6 *) |
| `compuerta` | 73,3 (+0,6) | 94,9 (-0,1) | 10,7 (+5,1 *) | 52,8 (+16,2 *) | 65,7 (+6,4 *) | 38,1 (+0,0) | 81,9 (+0,0) | 72,3 (-0,2) | 36,0 (+0,1) |
<!-- /T -->

- **Pesos fijos: no hay un valor bueno.** Las vagas quieren 0,1 para las palabras; las de
  siempre, 0,5 (con 0,1 pierden 7,8 puntos en primera); NFCorpus pierde con cualquiera.
- **Tratar distinto los términos ausentes** (`regla-fuera`: no cuentan; `regla-cero`:
  cuentan como corrientes; `regla-max-cob`: escalar por la fracción presente): arreglan las
  vagas a medias o del todo, pero a costa de las de siempre (`regla-cero` −4,9 en primera) o
  sin llegar (`regla-fuera`, 45,7 %).
- **Un tope de 0,45 o 0,4 a las palabras** (`regla-tope-*`): recupera las vagas (50,6 % y
  51,5 %) sin daño significativo en las de siempre ni en SciFact, pero **pierde en NFCorpus**
  (−0,6, de −1,1 a −0,2). Negativo como regla general.
- **Negativo de fondo:** con las estadísticas del vocabulario no se distingue una consulta
  vaga de una que no lo es. La fracción de la consulta que cubre el mejor acierto por
  palabras es la misma en los dos casos (mediana 0,24 en las de siempre, 0,23 en las vagas),
  porque las de siempre son largas (15,8 términos) y las vagas cortas (6,2).

**La regla que sí funciona: mirar la evidencia, no el vocabulario** (`lexical_gate=True`).
Lo que separa los dos tipos de consulta es el **BM25 del mejor acierto por palabras**, que
FTS5 ya devuelve gratis en la primera fila:

| Señal del mejor acierto por palabras (mediana; p10–p90) | siempre | vagas | SciFact | NFCorpus | MIRACL |
|---|---|---|---|---|---|
| Fuerza = BM25 / ln N («cuántas palabras raras casa») | 2,25 (1,33–4,26) | 0,99 (0,82–1,43) | 3,23 (1,90–6,03) | 1,42 (0,90–2,49) | 1,90 (1,46–2,56) |
| Cobertura = BM25 / Σ IDF de la consulta («qué parte de lo pedido casa») | 0,23 (0,14–0,46) | 0,24 (0,17–0,36) | 0,91 (0,51–1,00) | 1,00 (0,52–1,00) | 0,80 (0,63–0,99) |

La evidencia es floja solo cuando **las dos** son bajas: una consulta de varias palabras cuyo
mejor acierto comparte una o dos palabras corrientes. Cada señal es una rampa (fuerza de 1,0
a 2,0; cobertura de 0,4 a 0,6); la mayor de las dos escala el peso de las palabras entre 0,1
y el que le hubiera dado la regla de siempre, y lo que se quita va a los vectores.

- La primera versión solo miraba la fuerza. **Falló en NFCorpus** (−0,9, de −1,5 a −0,4):
  una consulta de una sola palabra rara tiene fuerza baja por construcción aunque case
  entera. Por eso se añadió la cobertura. Dicho claro: NFCorpus se usó para descubrir ese
  fallo, así que para esta regla ya no es un control limpio; sí lo es MIRACL, que se midió
  después y no participó en ningún ajuste.
- Sensibilidad. Los límites de la fuerza se fijaron a ojo sobre las notas; desplazarlos
  (1,25–1,75; 1,0–1,5; 1,5–2,5, medido con el prototipo de solo fuerza `regla-fuerza-*`) deja
  las vagas entre 20 de 51,9 % a 53,1 % y la primera de las de siempre de 72,2 % a 73,3 %.
  Los de la cobertura (0,4–0,6) se pusieron entre el p90 de las vagas (0,36) y el p10 de
  NFCorpus (0,52), es decir, **después de ver NFCorpus**; con 0,3–0,5, 0,5–0,7 y 0,4–0,8 la
  configuración recomendada no cambia más de 0,6 puntos en ninguno de los cuatro bancos
  ligeros (NFCorpus 36,0–36,1; vagas entre 20 54,7–55,3). En MIRACL no se repitió.
- Cuánto actúa: en el 99 % de las vagas (cierra del todo en el 53 %; peso medio de las
  palabras de 0,53 a 0,15), en el 40 % de las de siempre pero poco (0,49 → 0,41), en el 12 %
  de NFCorpus, el 7 % de SciFact y el 5 % de MIRACL.

**Veredicto: positivo, detrás de una opción.** Sola (`compuerta`): vagas entre 20 de 36,5 %
a 52,8 % y primera de 5,6 % a 10,7 %; siempre +0,6 en primera (de −0,4 a +1,7); MIRACL
idéntico; SciFact −0,2 (de −0,7 a 0,0); NFCorpus +0,1. No pierde en ninguno.

**Lo que no arregla.** En MIRACL la compuerta no hace nada porque allí el problema es el
contrario (palabras con demasiado poco peso); eso lo mejora quitar las palabras vacías
(experimento 1c), que sube la IDF media. Y con pesos fijos 0,6/0,4 más compuerta
(`rec+sw/sin-adapt`) NFCorpus vuelve a perder «entre 10»: la regla adaptativa sigue haciendo
falta para las consultas cortas de palabras raras.

**Coste.** Ninguno medible: reutiliza la primera fila de la búsqueda por palabras y la IDF ya
cacheada (2,80 ms con y sin compuerta en las notas; 8,1 frente a 8,3 ms en SciFact).

## 3. Corte por distancia

**Qué hace hoy.** Descarta los candidatos por vectores cuya distancia supera 1,3225 veces la
del mejor. Con MiniLM deja pasar de media 19,7 de 50 en las consultas de siempre y 29,5 en
las vagas. En el híbrido no reduce cuántos resultados se devuelven (las palabras rellenan
hasta `top_k`): lo que hace es quitar candidatos buenos por vectores y dejar el hueco a
candidatos por palabras.

**Valores más flojos** (sobre la regla con tope, para aislar el corte; y solo vectores):

<!-- T: corte2 tope45-corte1.3225 tope45-corte1.5 tope45-corte2 tope45-corte25 vec vec-corte1.5 vec-corte2 vec-corte25 -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `corte2` | 73,2 (+0,4) | 95,6 (+0,6) | 4,2 (-1,4 *) | 34,9 (-1,7) | 60,7 (+1,4) | 37,8 (-0,3) | 84,6 (+2,6 *) | 73,3 (+0,8 *) | 36,2 (+0,3) |
| `tope45-corte1.3225` | 72,1 (-0,7) | 95,3 (+0,3) | 7,1 (+1,5 *) | 50,6 (+14,0 *) | 63,5 (+4,2 *) | — | — | 72,5 (+0,0) | — |
| `tope45-corte1.5` | 72,6 (-0,1) | 96,0 (+1,0) | 5,7 (+0,1) | 49,9 (+13,3 *) | 68,1 (+8,8 *) | — | — | 73,0 (+0,5 *) | — |
| `tope45-corte2` | 72,2 (-0,6) | 96,1 (+1,1 *) | 4,9 (-0,7) | 46,2 (+9,7 *) | 69,4 (+10,1 *) | — | — | 73,3 (+0,8 *) | 35,5 (-0,4) |
| `tope45-corte25` | 72,2 (-0,6) | 96,1 (+1,1 *) | 4,7 (-0,8) | 46,0 (+9,4 *) | 69,4 (+10,1 *) | — | — | 73,4 (+0,9 *) | — |
| `vec` | 55,6 (-17,2 *) | 89,2 (-5,8 *) | 16,0 (+10,4 *) | 52,2 (+15,7 *) | 62,5 (+3,2 *) | 32,9 (-5,2 *) | 69,0 (-13,0 *) | 69,6 (-2,9 *) | 30,7 (-5,2 *) |
| `vec-corte1.5` | 55,6 (-17,2 *) | 91,1 (-3,9 *) | 16,0 (+10,4 *) | 55,8 (+19,3 *) | 71,7 (+12,4 *) | — | — | 71,1 (-1,4) | — |
| `vec-corte2` | 55,6 (-17,2 *) | 91,2 (-3,8 *) | 16,0 (+10,4 *) | 56,8 (+20,3 *) | 74,4 (+15,1 *) | 32,9 (-5,2 *) | 81,5 (-0,5) | 71,9 (-0,6) | 33,6 (-2,3 *) |
| `vec-corte25` | 55,6 (-17,2 *) | 91,2 (-3,8 *) | 16,0 (+10,4 *) | 56,8 (+20,3 *) | 74,4 (+15,1 *) | — | — | 72,0 (-0,5) | — |
<!-- /T -->

- **Aflojar el corte sube el recuerdo en todos los bancos**: «entre 50» en las vagas, «entre
  20» en MIRACL (+2,6 solo con `distance_cutoff=2.0`) y NDCG@10 en SciFact (+0,8, de +0,3 a
  +1,4). A partir de 2,0 ya no cambia nada (2,0, 3,0 y 25 dan lo mismo): con MiniLM y con
  bge-small equivale a no cortar.
- **Negativo, y es el hallazgo importante:** con los pesos de hoy, aflojar el corte
  **empeora los primeros puestos de las vagas** (primera de 7,1 % a 4,9 %; entre 5 de 26,4 %
  a 16,2 % sobre la regla con tope; con los pesos por defecto, `corte2` baja primera de 5,6
  a 4,2). La fusión RRF premia estar en las dos listas: al dejar pasar más candidatos por
  vectores, una nota mediocre que es la primera por palabras y la 45.ª por vectores suma
  0,0075 + 0,0052 y adelanta a la buena, tercera por vectores y ausente por palabras
  (0,0087). El corte estaba haciendo, de rebote, de freno a ese efecto.
- Por eso el corte flojo **solo conviene junto con la compuerta**, que baja el peso de las
  palabras cuando no son de fiar: `compuerta-corte2` da en las vagas primera 9,0 %, entre 20
  56,4 % y entre 50 74,2 %.

**Una versión que se adapta a la consulta: un suelo** (`min_vec_keep=N`: el corte nunca deja
menos de N candidatos por vectores).

<!-- T: suelo20 suelo compuerta-suelo20 compuerta-suelo compuerta-corte2 -->
| Configuración | siempre 1ª | siempre en 20 | vagas 1ª | vagas en 20 | vagas en 50 | MIRACL 1ª | MIRACL en 20 | SciFact NDCG@10 | NFCorpus NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| `suelo20` | 72,9 (+0,1) | 95,0 (+0,0) | 4,9 (-0,7) | 38,2 (+1,7 *) | 61,9 (+2,6 *) | 38,1 (+0,0) | 82,7 (+0,8) | 73,2 (+0,7 *) | 36,3 (+0,4) |
| `suelo` | 73,2 (+0,4) | 95,1 (+0,1) | 4,3 (-1,2 *) | 37,2 (+0,7) | 64,9 (+5,6 *) | — | — | 73,4 (+0,9 *) | 36,2 (+0,3) |
| `compuerta-suelo20` | 73,6 (+0,8) | 94,9 (-0,1) | 9,0 (+3,5 *) | 56,4 (+19,9 *) | 68,6 (+9,3 *) | — | — | 73,0 (+0,5) | 36,4 (+0,5 *) |
| `compuerta-suelo` | 73,9 (+1,1) | 95,4 (+0,4) | 9,0 (+3,5 *) | 56,5 (+20,0 *) | 74,3 (+15,0 *) | — | — | 73,1 (+0,6) | 36,3 (+0,4) |
| `compuerta-corte2` | 73,9 (+1,1) | 96,1 (+1,1) | 9,0 (+3,5 *) | 56,4 (+19,9 *) | 74,2 (+14,9 *) | 38,0 (-0,2) | 84,4 (+2,5 *) | 73,1 (+0,6) | 36,3 (+0,4) |
<!-- /T -->

- Funciona, pero **no mejora al corte flojo de toda la vida**: `compuerta-suelo` (suelo de
  50) da lo mismo que `compuerta-corte2`, y con suelo de 20 se queda corto en «entre 50»
  (68,6 % frente a 74,2 %). Veredicto: se queda como opción porque es barata y expresa mejor
  la intención («dame al menos N»), pero la recomendación usa `distance_cutoff=2.0`.
- No se probó un corte por distancia absoluta ni uno calibrado por modelo: el corte es una
  proporción respecto al mejor y cada modelo tiene su escala; habría que ajustarlo por
  modelo, y con los dos de aquí ya equivale a no cortar a partir de 2,0.

**Recuento de fallos antes y después** (`miss_analysis` con `search_options`, pidiendo 20):

| Consultas | Configuración | Fallos | Se cae en el corte | Por debajo del puesto 20 | No aparece | Estaba entre las 20 primeras por vectores |
|---|---|---|---|---|---|---|
| vagas | por defecto | 461 | 193 | 224 | 44 | 174 |
| vagas | recomendada | 322 | 0 | 276 | 46 | 15 |
| siempre | por defecto | 38 | 19 | 17 | 2 | 11 |
| siempre | recomendada | 27 | 0 | 25 | 2 | 3 |

Los 174 casos en que la fusión sacaba una nota que los vectores tenían entre las 20 primeras
bajan a 15.

**¿Y «solo cuando son relevantes»?** El corte es lo que hace que la búsqueda solo por
vectores devuelva pocas notas (19,7 de 50). Aflojarlo quita ese filtro: con 2,0 devuelve 45,5.
En el híbrido nunca filtró (devuelve 50 de 50 con cualquier valor). Para callar cuando no hay
nada relevante hace falta otra señal (el `min_relevance` de #521 o el reordenador); el corte
por distancia no es esa herramienta, y usarlo así cuesta recuerdo.

**Coste.** Ninguno.

## 4. Un modo de búsqueda «a fondo» (unir híbrido y solo vectores)

**Hipótesis (de rotten).** Juntar los candidatos del híbrido y los de solo vectores con corte
2,0 mete la nota buena más veces: 59,3 % → 76,9 % en las vagas con 50 + 50.

**Medida.** Se reproduce: 76,9 %. Pero la comparación justa es a igual número de candidatos:

| Consultas vagas | Candidatos | La buena entra | Primera · entre 20 | ms |
|---|---|---|---|---|
| Híbrido por defecto, 50 | 47,7 | 59,3 % | 5,6 · 36,5 | 1,4 |
| Unión: híbrido 50 + vectores 50 | 75,2 | 76,9 % | 5,6 · 52,2 | 2,2 |
| Híbrido por defecto, 100 | 89,9 | 69,7 % | 5,6 · 37,6 | 1,8 |
| Híbrido con corte 2,0, 100 | 99,7 | 80,4 % | 4,2 · 35,7 | 1,9 |
| **Compuerta + corte 2,0, 50** | 49,9 | **74,2 %** | 9,0 · 56,4 | 1,5 |
| **Compuerta + corte 2,0, 100** | 99,7 | **85,7 %** | 8,9 · 56,0 | 1,8 |
| Solo vectores con corte 2,0, 100 | 98,6 | 84,7 % | 16,0 · 56,8 | 1,0 |

| Consultas de siempre | Candidatos | La buena entra | Primera · entre 20 |
|---|---|---|---|
| Híbrido por defecto, 50 | 50,0 | 97,6 % | 72,8 · 95,0 |
| Unión: híbrido 50 + vectores 50 | 76,2 | 98,1 % | 72,8 · 95,4 |
| Compuerta + corte 2,0, 50 | 50,0 | 97,5 % | 73,9 · 96,1 |
| Compuerta + corte 2,0, 100 | 100,0 | 98,9 % | 73,9 · 96,0 |

<!-- UNION-MIRACL -->

**Veredicto: no merece ser una opción de `search`.** Lo que aportaba la unión era el corte
flojo de su segunda pata. Una sola búsqueda con compuerta y corte 2,0 mete la buena casi
igual con 50 candidatos (74,2 % frente a 76,9 % con 75) y bastante más con 100 (85,7 %), y
además deja la lista mejor ordenada (entre 20: 56,4 % frente a 52,2 %). La unión hecha con
dos llamadas cuesta un 35–60 % más de tiempo (2,2 frente a 1,4 ms en las vagas; 11,1 frente
a 8,1 ms en SciFact) y obliga a decidir fuera cómo intercalar las dos listas.

**Si lo que se quiere es recuerdo para un reordenador**: pedir más (`top_k=100`) a la
configuración recomendada. Es una búsqueda, cuesta 1,8 ms en vez de 1,5, y sube «la buena
entra» de 74 % a 86 % en las vagas.

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
todas las órdenes y subórdenes (31) construyen su `--help` con el typer instalado, que ningún módulo de la
CLI pasa `click_type` cuando typer no lo acepta, y que las tres opciones de elección siguen
rechazando valores desconocidos.

## Qué llevar a vstash

**De verdad (sin opción, no cambia resultados):**
1. El arreglo de la etiqueta de `miss_analysis` y `search_options` (experimento 5).
2. El test de compatibilidad de la CLI con typer (experimento 6).
3. Que el almacén lea del índice su tokenizador en vez de tenerlo escrito en dos sitios.

**Detrás de una opción, como están en la rama (apagadas por defecto):**
4. `lexical_gate` — el cambio con más efecto y sin coste. Candidata a valor por defecto más
   adelante, pero antes hay que pasarla por los cinco BEIR del artículo (aquí solo SciFact y
   NFCorpus) y por los modelos publicados v2/v3.
5. `drop_stopwords` — positivo junto con la compuerta y más rápido; sola empeora las
   consultas vagas, así que no debe activarse suelta.
6. `distance_cutoff=2.0` — ya existía. Sola mejora el recuerdo en MIRACL y en SciFact y no
   mueve las consultas de siempre ni NFCorpus, pero empeora los primeros puestos de las
   consultas vagas (primera 5,6 → 4,2), así que va con la compuerta. El
   valor por defecto (1,3225) cuesta 0,8 puntos de NDCG@10 en SciFact; merece revisarse con
   el BEIR completo, pero cambiarlo mueve los números publicados.
7. `fts_tokenizer="porter unicode61 remove_diacritics 2"` y `rebuild_fts` — ganancia
   pequeña y real en español (MIRACL), cero efecto en inglés. Buen candidato a valor por
   defecto para índices nuevos en una versión mayor.
8. `min_vec_keep` — funciona pero no mejora a `distance_cutoff=2.0`. Se puede dejar o quitar.

**No llevar:**
9. El modo «a fondo» por unión: redundante con 4 + 6.
10. Trigramas y `unicode61` sin raíces: pierden en inglés y no ganan en español.
11. Topes o pesos fijos para la fusión: cada uno pierde en algún banco.
<!-- CIERRE-SNOWBALL -->

**Falta por cablear si se adopta:** las opciones están en `VstashStore.search`,
`services.search_with_embedding` y `Memory.search`. No están en la CLI, MCP, LangChain,
`federated_search`, `vstash.toml` ni `docs/`. `miss_analysis` las acepta por
`search_options`; `vstash why` no.

## Lo que quedó sin verificar

- **Inglés:** solo SciFact y NFCorpus. FiQA, SciDocs y ArguAna no se pasaron (ArguAna usa la
  rama de consultas largas, que estas opciones no tocan, pero no se midió). Tampoco los
  modelos afinados v2/v3.
- **Consultas reales de Jay:** las vagas son un extremo artificial y las de siempre el otro;
  MIRACL son preguntas a Wikipedia. No hay consultas reales medidas.
- **Un trozo por documento** en todos los bancos: no se midió la interacción de la compuerta
  con MMR ni con documentos de muchos trozos.
- **Otros backends de vectores** (snapvec, IVFPQ): no probados; `min_vec_keep` y la
  compuerta no dependen de ellos, pero no hay medida.
- `rebuild_fts` tiene tests de corrección pero no se midió su tiempo ni se probó sobre una
  base grande real. Los trigramas no se midieron en MIRACL.
- La compuerta se ajustó con las notas y se corrigió mirando NFCorpus; su control limpio es
  MIRACL, donde casi no actúa (5 % de las consultas). Falta un banco de consultas vagas que
  no sea el inventado.
- Los intervalos son de remuestreo sobre las consultas de cada banco; no hay repetición con
  otras semillas de datos ni corrección por haber mirado muchas configuraciones.

## Coste del laboratorio

Todo en local, sin LLM ni API. Lo más caro fue vectorizar MIRACL una vez (146.750 pasajes,
unos 19 minutos de reloj con nueve núcleos ocupados, 3,3 GB de memoria en el pico) y cada pasada de MIRACL
(648 consultas, 1–3 minutos). Los vectores y los índices quedan en
`experiments/data/lab_es/` (ignorado por git, ~1,5 GB; se puede borrar).
