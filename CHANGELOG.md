# Changelog

Reporte final de la actividad y registro de versiones del repositorio y del dataset.

## [0.3.0] - 2026-10-18 · datos_ini

### Agregado
- `fuentes/obtener_datos.py`: manifiesto de las 7,012 imágenes del origen en un commit fijo, descarga a la caché de Hugging Face y verificación sha256.
- `datos/datos_ini/manifiesto.csv`.
- `requirements.txt`: dependencias de obtención y preparación de datos.

### Corregido
- La descarga con `snapshot_download(allow_patterns=...)` tardaba más de 15 minutos solo en filtrar 2,662 rutas exactas; ahora se pide archivo por archivo en paralelo (12 s, y casi instantáneo con la caché).

## [0.2.0] - 2026-10-18

### Agregado
- Configuración de MLflow con seguimiento local (`mlflow.db` + `mlruns/` en la raíz del repo) en `fuentes/train.py`.
- `python fuentes/train.py --validar`: corrida de prueba que registra parámetros, métricas, artefactos y modelo, y verifica cada registro releyéndolo de MLflow.
- Etiquetas de trazabilidad en cada corrida: `git_commit` y `git_cambios_sin_commit`.
- `requirements.txt` con versiones fijas.

### Corregido
- Requisitos del modelo declarados de forma explícita; la inferencia automática de MLflow incluía torch, dill y tqdm, que el modelo no usa.

## [0.1.0] - 2026-10-18

### Agregado
- Estructura inicial del repositorio: `datos/datos_ini`, `datos/datos_limp`, `fuentes/` (`entrena.ipynb`, `datos_prep.py`, `train.py`), `README.md` y `CHANGELOG.md`.
- `.gitignore` para archivos temporales de Python y Jupyter.
