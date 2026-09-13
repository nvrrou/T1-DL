# Material de entrega (secciones 1 y 2)

Completar con los resultados de la ejecucion definitiva. Este documento no
contiene conclusiones experimentales ni sustituye la presentacion y el video.

- [ ] Dataset: fuente, licencia, naturaleza simulada, filas seleccionadas y
      retenidas, numero de features y distribucion de las cinco clases.
      Fuente: `archive/data/clean/artifacts/dataset_manifest.json`.
- [ ] Baselines: resultados de al menos dos modelos clasicos; limitaciones
      sustentadas por curvas, matrices de confusion y errores por categoria.
      Distinguir SVM RBF (muestra) de la comparacion principal con train completo.
- [ ] DNN: justificar capas, neuronas, embeddings, activacion, dropout,
      optimizador, tasa de aprendizaje, pesos de clase y early stopping.
- [ ] Convergencia: incorporar `08_dnn_convergencia.png` y examinar
      `dnn_history.csv`; explicar los comportamientos observados.
- [ ] AutoML: herramienta, candidatos, presupuesto y modelo seleccionado.
- [ ] NAS: espacio de arquitecturas, trials, pruning, presupuesto,
      arquitectura seleccionada y reentrenamiento final.
- [ ] Comparacion: desempeno, complejidad, costo y generalizacion. Incorporar
      `comparacion_modelos.csv` y `14_comparacion.png`; considerar hardware,
      tiempos de busqueda/entrenamiento y tamano serializado de cada modelo.
- [ ] Discusion propia: responder si se justifica una DNN y si el diseno manual
      aporto alguna ventaja respecto de AutoML y NAS. Basarse en los numeros
      obtenidos, sin asumir de antemano que la DNN debe ganar.
- [ ] Adjuntar codigo, requirements, instrucciones, graficos y logs.
- [ ] Preparar diapositivas y grabar un video de un maximo de 20 minutos.
- [ ] Declarar especificamente el uso de herramientas generativas. El
      enunciado y la rubrica prohiben su uso para el diseno de soluciones y
      el analisis de resultados; documentar fielmente la asistencia recibida.

No se incluye la seccion 3 (generacion y aumento con datos sinteticos).
