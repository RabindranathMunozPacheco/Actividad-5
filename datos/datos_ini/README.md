# datos_ini

Datos originales, tal como se obtuvieron de la fuente. No se modifican; cualquier corrección se hace con `fuentes/datos_prep.py` y el resultado se guarda en `datos/datos_limp/`.

- `manifiesto.csv`: una fila por imagen del repositorio [`muAtarist/maize_foliar_diseases`](https://huggingface.co/datasets/muAtarist/maize_foliar_diseases) en el commit `448bec6c7abe18fe87917e443368c63ed3786119` (7,012 imágenes). Columnas: `ruta`, `carpeta`, `archivo`, `bytes`, `sha256`.

Las imágenes no se guardan aquí porque el repositorio de origen no declara licencia. `python fuentes/obtener_datos.py` regenera el manifiesto, descarga a la caché local de Hugging Face las 2,662 imágenes que usa la limpieza y comprueba que el sha256 de cada una coincide con el del manifiesto.
