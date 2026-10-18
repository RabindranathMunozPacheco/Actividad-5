# Changelog

Reporte final de la actividad y registro de versiones del repositorio y del dataset.

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
