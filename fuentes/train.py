"""Entrenamiento con Grid Search y validación cruzada, y registro en MLflow.

Lee los datos de datos/datos_limp/ y registra parámetros, métricas y modelos
de cada algoritmo como corridas de un mismo experimento de MLflow.

Uso:
    python fuentes/train.py --validar   # corrida de prueba de la configuración de MLflow
"""
import argparse
import os
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

# Antes de importar mlflow: sin barra de progreso ni avisos de agentes en la consola
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo se guardan las figuras
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

RAIZ = Path(__file__).resolve().parents[1]
URI_LOCAL = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
DIR_ARTEFACTOS = RAIZ / "mlruns"
SEMILLA = 42
# Requisitos explícitos del modelo: si se dejan a la inferencia de MLflow, registra todo
# lo que encuentra cargado en el entorno (torch, dill, tqdm...) aunque el modelo no lo use.
REQUISITOS_MODELO = [f"{p}=={version(p)}" for p in ("scikit-learn", "numpy", "pandas", "scipy", "skops")]

# Aviso informativo de MLflow sobre columnas enteras en la firma; las clases son enteras a propósito
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def configurar_mlflow(experimento):
    """Apunta MLflow a la base del repo (o a MLFLOW_TRACKING_URI) y activa el experimento.

    La ruta se fija a la raíz del repo para que todas las corridas queden en el mismo
    mlflow.db sin importar desde qué carpeta se ejecute el script.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", URI_LOCAL)
    mlflow.set_tracking_uri(uri)
    if uri == URI_LOCAL and mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=(DIR_ARTEFACTOS / experimento).as_uri())
    mlflow.set_experiment(experimento)
    return uri


def version_git():
    """Commit actual del repo y si hay cambios sin commit, para la trazabilidad de cada corrida."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD") or "sin_git",
            "git_cambios_sin_commit": str(bool(git("status", "--porcelain")))}


def validar_mlflow():
    """Corrida mínima con iris que registra parámetros, métricas, artefactos y modelo, y los relee."""
    uri = configurar_mlflow("validacion-mlflow")
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
    params = {"modelo": "LogisticRegression", "C": 1.0, "max_iter": 500, "semilla": SEMILLA,
              "dataset": "iris (scikit-learn)", "test_size": 0.2}
    modelo = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=SEMILLA)

    with mlflow.start_run(run_name="prueba-configuracion") as run:
        mlflow.set_tags({**version_git(), "proposito": "validar registro de MLflow (paso 2)"})
        mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
        f1_folds = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_macro")
        for i, f1 in enumerate(f1_folds):
            mlflow.log_metric("cv_f1_macro_fold", f1, step=i)
        mlflow.log_metric("cv_f1_macro_media", f1_folds.mean())

        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        mlflow.log_metrics({"test_accuracy": accuracy_score(y_te, pred),
                            "test_f1_macro": f1_score(y_te, pred, average="macro")})

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, colorbar=False)
        mlflow.log_figure(fig, "figuras/matriz_confusion.png")
        plt.close(fig)
        mlflow.log_dict(params, "config/parametros.json")

        info = mlflow.sklearn.log_model(modelo, name="modelo", input_example=X_te.head(5),
                                        pip_requirements=REQUISITOS_MODELO)

    # Releer desde la base de MLflow lo que se acaba de registrar
    corrida = MlflowClient().get_run(run.info.run_id)
    artefactos = [a.path for a in MlflowClient().list_artifacts(run.info.run_id, "figuras")
                  + MlflowClient().list_artifacts(run.info.run_id, "config")]
    recargado = mlflow.sklearn.load_model(info.model_uri)
    requisitos = Path(mlflow.pyfunc.get_model_dependencies(info.model_uri)).read_text()
    revisiones = {
        "parámetros": set(map(str, params)) <= set(corrida.data.params),
        "métricas": {"cv_f1_macro_media", "test_accuracy", "test_f1_macro"} <= set(corrida.data.metrics),
        "métrica por fold (5 pasos)": len(MlflowClient().get_metric_history(run.info.run_id, "cv_f1_macro_fold")) == 5,
        "artefactos (figura y json)": {"figuras/matriz_confusion.png", "config/parametros.json"} <= set(artefactos),
        "modelo recargable y predice igual": np.array_equal(recargado.predict(X_te), pred),
        "requisitos del modelo sin paquetes ajenos": "torch" not in requisitos,
        "tag git_commit": "git_commit" in corrida.data.tags,
    }

    print(f"\nSeguimiento: {uri}")
    print(f"Experimento: validacion-mlflow | corrida: {run.info.run_id}")
    print(f"Modelo: {info.model_uri}\n")
    for nombre, ok in revisiones.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {nombre}")
    print(f"\nMétricas: { {k: round(v, 4) for k, v in corrida.data.metrics.items()} }")
    return all(revisiones.values())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # acentos legibles en la consola de Windows
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validar", action="store_true", help="corrida de prueba de la configuración de MLflow")
    args = parser.parse_args()
    if args.validar:
        raise SystemExit(0 if validar_mlflow() else 1)
    parser.print_help()
