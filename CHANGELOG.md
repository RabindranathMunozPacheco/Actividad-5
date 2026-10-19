# Changelog

Reporte final de la actividad y registro de versiones del repositorio y del dataset.

## [0.4.0] - 2026-10-18 · datos_limp

### Agregado
- `fuentes/datos_prep.py`: 8 reglas de limpieza (R1 a R8) con registro por imagen, estandarización a RGB 224×224, deduplicación exacta (sha256) y perceptual (dHash), grupos de fotos de la misma planta y sesión de captura.
- `datos/datos_limp/`: `registro_limpieza.csv`, `maiz_parcela.csv` (1,078 fotos), `caracteristicas.parquet` (1,280 por foto) y `referencia_plantvillage.parquet`.
- Corrida `preparacion` en el experimento `maiz-datos` de MLflow con la etiqueta `datos_sha256`, que también llevan las corridas de entrenamiento.
- `.gitattributes`: los CSV siempre con fin de línea LF, para que `datos_sha256` sea igual en cualquier sistema operativo (con `core.autocrlf` un clon en Windows los cambiaba a CRLF y cambiaba su huella).
- `fuentes/train.py`: regresión logística y random forest con Grid Search y `StratifiedGroupKFold` agrupado por fotos de la misma planta; una corrida hija en MLflow por cada combinación de hiperparámetros; prueba final de 20 %. Entra en esta versión porque `datos_prep.py` usa su configuración de MLflow.
- Comparación con 5 × 5 folds pareados, t corregida de Nadeau y Bengio y Wilcoxon; evaluación dejando fuera sesiones de captura completas (`LeaveOneGroupOut`); registro del modelo elegido en el Model Registry.

### Cambiado
- Rejilla del random forest: `min_samples_leaf` ampliado de {1, 3, 5} a {1, 3, 5, 10, 20} porque la primera rejilla dejó el óptimo en el borde.
- `git_cambios_sin_commit` solo considera cambios en código, datos y dependencias; la documentación no altera los resultados.

### Corregido
- Procesador de imágenes con backend PIL explícito: `MobileNetV2ImageProcessor` cambia de backend si torchvision está instalado y las características cambiarían de un equipo a otro.
- El random forest no se podía registrar: skops rechaza por seguridad el tipo `sklearn.tree._tree.Tree`; se declara confiable solo ese tipo y solo para ese modelo.
- El intervalo de confianza de la diferencia entre modelos se calculaba sin la corrección de Nadeau y Bengio y no incluía el cero, en contradicción con la prueba; ahora usa el mismo error estándar corregido.

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
