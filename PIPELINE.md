# Pipeline de preprocesamiento — Tarea 1 DP

Pipeline para seleccionar aproximadamente 3 millones de registros del dataset
E-Commerce Price Tracker y resolver una clasificacion multiclase con al menos
15 caracteristicas. La limpieza y el preprocesamiento trabajan por bloques.

## Orden de ejecucion

El objetivo es multiclase a 7 dias. Se calcula como
`100 * (target_price_7d - price) / price`, antes de escalar los datos:

| Clase | Cambio porcentual |
| --- | --- |
| `0_baja_mucho` | <= -10% |
| `1_baja_un_poco` | > -10% y < -3% |
| `2_se_mantiene` | de -3% a +3%, inclusive |
| `3_sube_un_poco` | > +3% y < +10% |
| `4_sube_mucho` | >= +10% |

Los umbrales se configuran en `target_config.py`. `02_feature_engineering.py`
reemplaza el target original de tres clases con estas cinco etiquetas.
Los precios futuros se excluyen de las entradas de todos los modelos.
Se descartan filas sin precios validos para calcular el cambio.
DNN y NAS infieren las cinco salidas desde las etiquetas. La regresion lineal
conserva su adaptacion original: redondear y limitar la prediccion al indice de
clase (0 a 4).

En PowerShell, desde `F:\T1DL`, instala las dependencias una vez:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El paso 00 no forma parte de la ejecucion necesaria. Regenera los datos con 01
y 02 y vuelve a entrenar todos los modelos: los resultados anteriores no
corresponden a las cinco clases ni al nuevo split temporal.

```
.\run_pipeline.ps1               # Ejecuta 01 a 07 y guarda logs
.\run_pipeline.ps1 -Desde 3 -Hasta 4  # Ejecuta solo baseline y DNN
python 05b_automl_robustez.py     # Opcional, despues de 05
```

Si la politica de PowerShell bloquea scripts, o si se quieren revisar los
resultados de cada etapa antes de continuar, ejecutar uno por uno:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py 01_limpieza_datos.py
& $py 02_feature_engineering.py
& $py 03_baselines.py
& $py 04_dnn.py
& $py 05_automl.py
& $py 06_nas.py
& $py 07_comparacion.py
```

Modulos de la DNN (usados por 04 y 06): `dnn_config.py`, `dnn_data.py`,
`dnn_model.py`, `dnn_plots.py`.

## Fases

### Limpieza canonica (`01_limpieza_datos.py`)
Motor out-of-core (**DuckDB**). Registra los 3 CSV como vistas virtuales,
elimina identificadores no predictivos (`product_id`, `title`, `brand`, `asin`,
`timestamp`), filtra precios invalidos y deduplica. Selecciona de forma
reproducible aproximadamente 3 millones de filas y separa cronologicamente
train/validacion/test. Purga siete dias entre splits para que el horizonte del
target no cruce sus limites. Exporta Parquet con categorias en texto y sin
escalar. `split_metadata.json` conserva fechas y cantidades.

```
archive/data/clean/{train,val,test}.parquet
```

### Feature Engineering (`02_feature_engineering.py`)
Transformaciones matematicas con arquitectura **Two-Pass Streaming**. Lee el
Parquet limpio de la Fase 2 por lotes (`pyarrow.iter_batches`).

**Pasada 1 (fit, solo TRAIN):**
- *Imputacion:* suma y conteo por columna numerica -> **media incremental**.
- *Categoricas:* vocabulario acumulado por columna via **`set.union`** entre
  chunks (mas frecuencias para la moda). `LabelEncoder.fit()` se llama sobre el
  vocabulario (pocos valores unicos), nunca sobre arrays masivos.
- *Escalado:* segundo barrido imputando cada bloque y llamando
  **`StandardScaler.partial_fit()`** bloque a bloque.

**Pasada 2 (transform & write, cada split):**
- Relee por chunks, imputa, codifica categoricas, aplica `scaler.transform()`.
- **Downcasting:** `float32` para features numericas, `int16` para codigos
  categoricos, `float32` para targets de regresion.
- Escribe cada bloque directo a un Parquet unificado con
  **`pyarrow.parquet.ParquetWriter(write_table)`** y libera RAM en cada
  iteracion (`del chunk; gc.collect()`).

Artefactos exportados en `archive/data/clean/artifacts/`:
`scaler.pkl`, `encoders.pkl`, `imputers.pkl`, `feature_names.pkl`.

Salida:
```
archive/data/clean/{train,val,test}_features.parquet
```

### Paso 1 — Modelos baseline (`03_baselines.py`)
Consume `*_features.parquet` de la Fase 3 y entrena regresion lineal, logistica
y SVM (lineal y RBF) para evidenciar limitaciones de los modelos lineales.
La regresion logistica principal entrena sobre el train completo para que sea
comparable con DNN, AutoML y NAS. SVM RBF queda identificado como experimento
secundario con una muestra de 50.000 filas por su costo cuadratico.

### Paso 2 — DNN (`04_dnn.py`)
MLP tabular en PyTorch. Decisiones de diseno:
- Categoricas por capas de **embedding** aprendidas (no codigos crudos ni
  one-hot); numericas ya estandarizadas entran directo.
- Bloques Linear -> BatchNorm -> ReLU -> Dropout; salida de 5 clases.
- **CrossEntropy ponderada** por frecuencia inversa (pesos calculados con las cinco clases del train).
- Adam + ReduceLROnPlateau + **early stopping** sobre la perdida de validacion.
- Entrenamiento por **mini-batches en streaming** desde el Parquet seleccionado.
- Auto-detecta GPU (CUDA) o CPU.

Salidas: `results/08_dnn_convergencia.png` (curvas train/val de loss y accuracy),
`results/09_dnn_confusion.png`, y el modelo en
`archive/data/clean/artifacts/dnn_model.pt`.

### Paso 3 (2.1) — AutoML (`05_automl.py`)
FLAML busca automaticamente una tecnica de ML de baja complejidad (LightGBM,
Random Forest, Extra Trees, Regresion Logistica), optimizando macro-F1 sobre
el train comun. Usa exclusivamente validacion durante la busqueda y reserva
test para la evaluacion final. Guarda métricas, reporte por clase y confusion.

### Paso 3 (2.2) — NAS (`06_nas.py`)
Optuna optimiza la arquitectura del MLP del paso 2 (capas, unidades, dropout, lr,
batch size). Cada trial entrena el MLP con embeddings sobre el train comun y
evalua en validacion; se reentrena la mejor, conserva su mejor epoca y solo
entonces se evalua en test. Guarda
`results/metrics_nas.json`, la historia de la busqueda y la confusion.

### Paso 3 (2.3) — Comparacion (`07_comparacion.py`)
Valida que las 4 soluciones pertenezcan exactamente al mismo procesamiento y
compara baseline principal, DNN manual, AutoML y NAS en el mismo test. Registra
accuracy, macro-F1, brecha train-test, parametros, tamaño serializado, tiempo de
busqueda, entrenamiento e inferencia, dispositivo y versiones. Luego
produce `results/comparacion_modelos.csv` y `results/14_comparacion.png` con
desempeno, complejidad, costo y generalizacion. El equipo redacta el analisis
critico a partir de estos numeros, de acuerdo con `ENTREGA.md`.

Dependencias del paso 3: `pip install "flaml[automl]" optuna`.

## Nota sobre la fuente del streaming en la Fase 3
La Fase 3 lee el **Parquet limpio de la Fase 2** en vez de releer el CSV crudo.
Es preferible porque ya viene deduplicado, sin filas sin target y con dtypes
reducidos; la lectura por lotes de Parquet cumple el mismo objetivo out-of-core
que un `read_csv(chunksize=...)` sobre el CSV original, sin reintroducir filas
invalidas.

## Alternativa de alto rendimiento (opcional)
Si no fuera obligatorio conservar scikit-learn para exportar artefactos `.pkl`,
la limpieza y transformacion podrian implementarse enteramente en Polars Lazy
(`pl.scan_csv()...collect(streaming=True)`) o DuckDB, aprovechando el
paralelismo multinucleo nativo. El pipeline actual usa DuckDB en la Fase 2 y
mantiene scikit-learn en la Fase 3 justamente para producir los artefactos
(`scaler.pkl`, `encoders.pkl`) que necesitan los modelos.

## Archivos retirados
`01_limpieza_colab.py` (monolito con `pd.concat` + muestreo) fue retirado; el
codigo original se conserva en `legacy/01_limpieza_colab.py`.
