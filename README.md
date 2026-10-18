# Actividad 5. Entrenamiento, ajuste y registro de modelos con MLflow

Gestión de proyectos de inteligencia artificial, Universidad Tecmilenio.

Guía de reproducción y registro de las decisiones tomadas en la preparación de datos, el entrenamiento de los modelos y su registro en MLflow.

## 1. Descripción del caso

_Pendiente: problema, dataset y variable objetivo._

## 2. Estructura del repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/        # datos originales, sin modificar
│   └── datos_limp/       # datos después de limpieza y deduplicación
├── fuentes/
│   ├── entrena.ipynb     # notebook de Colab: entrenamiento y evaluación
│   ├── datos_prep.py     # funciones de limpieza
│   └── train.py          # entrenamiento y registro en MLflow
├── README.md             # este archivo
├── CHANGELOG.md          # reporte final y versiones
└── requirements.txt      # dependencias con versión fija
```

Al ejecutar se crean `mlflow.db` y `mlruns/` en la raíz; no se suben al repositorio (ver sección 4).

## 3. Datos

_Pendiente: origen, licencia, diccionario de datos y versiones del dataset._

## 4. Cómo reproducir

### Entorno

- Python 3.10, Windows 11 (también corre en Colab).
- Dependencias: `pip install -r requirements.txt`

### Seguimiento con MLflow

Se usa **seguimiento local**: los parámetros, métricas y etiquetas van a una base SQLite (`mlflow.db`) y los artefactos (figuras, archivos y modelos) a la carpeta `mlruns/`, ambas en la raíz del repositorio. `train.py` fija esa ruta a la raíz para que todas las corridas queden en la misma base sin importar desde qué carpeta se ejecute. Para usar un servidor remoto basta con definir la variable de entorno `MLFLOW_TRACKING_URI`.

Cada corrida guarda como etiquetas el commit de Git (`git_commit`) y si había cambios sin commit (`git_cambios_sin_commit`), para saber con qué versión del código y de los datos se obtuvo cada resultado. Los modelos se registran con sus requisitos declarados de forma explícita, porque la inferencia automática de MLflow agregaba paquetes que el modelo no usa (por ejemplo, torch).

Validar la configuración (corrida de prueba con iris):

```bash
python fuentes/train.py --validar
```

Comprueba que se registran parámetros, métricas (incluida una métrica por fold), artefactos y el modelo, y que el modelo se puede recargar y predice igual.

Abrir la interfaz de MLflow (desde la raíz del repositorio) y entrar a http://127.0.0.1:5000:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

_Pendiente: orden de ejecución de limpieza y entrenamiento._

## 5. Decisiones

_Pendiente: criterios de limpieza, modelos, rejilla de hiperparámetros y métricas._

## 6. Resultados

_Pendiente: comparación de modelos, prueba estadística y experimentos en MLflow._

## 7. Deuda técnica y acciones MLOps

_Pendiente._
