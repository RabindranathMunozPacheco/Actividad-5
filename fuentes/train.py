"""Entrenamiento con Grid Search y validación cruzada, y registro en MLflow.

Lee el dataset limpio de datos/datos_limp/ (maiz_parcela.csv + caracteristicas.parquet), entrena
regresión logística y random forest con Grid Search, los compara con una prueba estadística y registra
todo en el experimento "maiz-logreg-vs-rf" de MLflow:
    - una corrida por modelo, con una corrida hija por cada combinación de hiperparámetros
    - una corrida "comparacion" con la validación cruzada repetida, las pruebas estadísticas y la
      evaluación dejando fuera sesiones de captura completas
    - el modelo elegido queda en el Model Registry como "clasificador-hoja-maiz" con el alias "candidato"
Las figuras y una tabla con todas las corridas se copian también a evidencias/.

Uso:
    python fuentes/train.py             # entrenamiento, comparación y registro
    python fuentes/train.py --validar   # corrida de prueba de la configuración de MLflow
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

# Antes de importar mlflow: sin barra de progreso ni avisos de agentes en la consola
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # en consola, sin ventana: solo se guardan las figuras
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, ParameterGrid, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
URI_LOCAL = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
DIR_ARTEFACTOS = RAIZ / "mlruns"
DATOS_LIMP = RAIZ / "datos" / "datos_limp"
EVIDENCIAS = RAIZ / "evidencias"
SEMILLA = 42
# Requisitos explícitos del modelo: si se dejan a la inferencia de MLflow, registra todo
# lo que encuentra cargado en el entorno (torch, dill, tqdm...) aunque el modelo no lo use.
REQUISITOS_MODELO = [f"{p}=={version(p)}" for p in ("scikit-learn", "numpy", "pandas", "scipy", "skops")]

EXPERIMENTO = "maiz-logreg-vs-rf"
MODELO_REGISTRADO = "clasificador-hoja-maiz"
CLASES = {"Sana": "sana", "Mancha gris": "mancha_gris", "Tizón": "tizon", "Roya común": "roya"}  # clave ASCII para métricas
N_FOLDS = 5
REPETICIONES = 5            # 5 x 5 folds = 25 puntajes pareados para la prueba estadística
ALFA = 0.05
METRICAS_CV = ["f1_macro", "balanced_accuracy", "accuracy"]

# Aviso informativo de MLflow sobre columnas enteras en la firma; las clases son enteras a propósito
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
# Aviso de MLflow 3.17 al registrar un dataset con ruta local: dos resolvedores iguales, elige el mismo tipo
warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted in multiple ways")


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
    """Commit actual del repo y si hay cambios sin commit, para la trazabilidad de cada corrida.

    Solo cuentan los cambios que pueden alterar un resultado (código, datos y dependencias), no los
    de la documentación ni las evidencias que el propio entrenamiento regenera.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    cambios = git("status", "--porcelain", "--", "fuentes/*.py", "datos", "requirements.txt")
    return {"git_commit": git("rev-parse", "HEAD") or "sin_git",
            "git_cambios_sin_commit": str(bool(cambios))}


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


# ---------------------------------------------------------------------------------------------
# Entrenamiento y comparación con el dataset de hojas de maíz
# ---------------------------------------------------------------------------------------------

def modelos():
    """Estimador, rejilla de hiperparámetros y paralelismo del Grid Search de cada algoritmo.

    Justificación de las rejillas en el README (sección 5). En resumen: hay más características (1,280)
    que fotos de entrenamiento, así que la regresión logística necesita regularización fuerte y C se
    explora en escala logarítmica; en random forest el número de árboles se fija (más árboles no
    sobreajustan, solo cuestan) y se ajusta lo que controla la complejidad de cada árbol.
    En las dos se prueba compensar el desbalance de clases (Mancha gris tiene casi 6 veces más fotos que Roya).
    """
    return {
        "regresion_logistica": (
            Pipeline([("escala", StandardScaler()),
                      ("modelo", LogisticRegression(max_iter=5000, random_state=SEMILLA))]),
            {"modelo__C": [0.0001, 0.001, 0.01, 0.1, 1.0],
             "modelo__class_weight": [None, "balanced"]},
            -1),                                   # Grid Search en paralelo: cada ajuste usa un núcleo
        "random_forest": (
            RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEMILLA),
            {"min_samples_leaf": [1, 3, 5, 10, 20],    # una primera rejilla hasta 5 dejó el óptimo en el borde
             "max_features": ["sqrt", 0.1, 0.25],
             "class_weight": [None, "balanced_subsample"]},
            1),                                    # el bosque ya usa todos los núcleos
    }


def huella(*rutas):
    """sha256 corto del contenido de los archivos de datos: identifica la versión del dataset."""
    h = hashlib.sha256()
    for ruta in rutas:
        h.update(ruta.read_bytes())
    return h.hexdigest()[:16]


def cargar_datos():
    """Dataset limpio unido con sus características, y la lista de columnas de características."""
    tabla = pd.read_csv(DATOS_LIMP / "maiz_parcela.csv")
    caract = pd.read_parquet(DATOS_LIMP / "caracteristicas.parquet")
    datos = tabla.merge(caract, on="id", how="inner", validate="one_to_one")
    return datos, [c for c in caract.columns if c != "id"]


def separar_prueba(datos):
    """Un fold de 5 (20 %) como prueba final; los grupos de fotos parecidas no se parten."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    ent, prueba = next(cv.split(datos, datos["clase"], datos["grupo"]))
    return datos.iloc[ent].reset_index(drop=True), datos.iloc[prueba].reset_index(drop=True)


def metricas(y, pred, prefijo):
    """Métricas globales, F1 por clase y la tasa de hojas enfermas clasificadas como sanas."""
    y, pred = np.asarray(y), np.asarray(pred)
    res = {f"{prefijo}_f1_macro": f1_score(y, pred, average="macro"),
           f"{prefijo}_balanced_accuracy": balanced_accuracy_score(y, pred),
           f"{prefijo}_accuracy": accuracy_score(y, pred),
           # el error más caro para el productor: una hoja enferma que la app reporta como sana
           f"{prefijo}_enferma_como_sana": float(np.mean(pred[y != "Sana"] == "Sana"))}
    for clase, f1 in zip(CLASES, f1_score(y, pred, labels=list(CLASES), average=None, zero_division=0)):
        res[f"{prefijo}_f1_{CLASES[clase]}"] = f1
    return res


def guardar_figura(fig, nombre):
    """Registra la figura en la corrida activa y la copia a evidencias/figuras/ para el README."""
    mlflow.log_figure(fig, f"figuras/{nombre}")
    (EVIDENCIAS / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(EVIDENCIAS / "figuras" / nombre, dpi=120, bbox_inches="tight")
    plt.close(fig)


def figura_matriz(y, pred, titulo):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=list(CLASES), normalize="true", values_format=".2f",
                                            ax=ax, colorbar=False, cmap="Greens")
    ax.set(title=titulo, xlabel="Predicción", ylabel="Clase real")
    return fig


def figura_rejilla(res, rejilla, nombre):
    """F1 macro de validación (y de entrenamiento, punteado) contra el primer hiperparámetro de la rejilla."""
    params = list(rejilla)
    eje, resto = params[0], params[1:]
    res = res.astype({f"param_{p}": str for p in params})        # None se agruparía como NaN
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [str(v) for v in rejilla[eje]]
    for valores, sub in res.groupby([f"param_{p}" for p in resto]):
        sub = sub.set_index(sub[f"param_{eje}"].astype(str)).loc[x]
        etiqueta = ", ".join(f"{p.split('__')[-1]}={v}" for p, v in zip(resto, valores))
        linea = ax.errorbar(x, sub["mean_test_f1_macro"], yerr=sub["std_test_f1_macro"], marker="o", capsize=3,
                            label=etiqueta)
        ax.plot(x, sub["mean_train_f1_macro"], ls=":", color=linea[0].get_color())
    ax.set(title=f"{nombre}: F1 macro en validación cruzada (punteado: entrenamiento)",
           xlabel=eje.split("__")[-1], ylabel="F1 macro")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    return fig


def entrenar_modelo(nombre, estimador, rejilla, n_jobs, ent, prueba, columnas, etiquetas):
    """Grid Search con validación cruzada agrupada, evaluación en prueba y registro completo en MLflow."""
    X, y, grupos = ent[columnas], ent["clase"], ent["grupo"]
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(estimador, rejilla, scoring=METRICAS_CV, refit="f1_macro", cv=cv,
                            n_jobs=n_jobs, return_train_score=True)

    with mlflow.start_run(run_name=nombre) as corrida:
        mlflow.set_tags({**etiquetas, "modelo": nombre, "etapa": "grid_search"})
        mlflow.log_input(mlflow.data.from_pandas(ent, source=str(DATOS_LIMP), name="maiz_parcela",
                                                 targets="clase"), context="entrenamiento")
        mlflow.log_input(mlflow.data.from_pandas(prueba, source=str(DATOS_LIMP), name="maiz_parcela",
                                                 targets="clase"), context="prueba")
        mlflow.log_params({"algoritmo": type(estimador[-1] if isinstance(estimador, Pipeline) else estimador).__name__,
                           "rejilla": json.dumps(rejilla), "n_combinaciones": len(ParameterGrid(rejilla)),
                           "validacion_cruzada": f"StratifiedGroupKFold({N_FOLDS}), grupos de fotos parecidas",
                           "metrica_seleccion": "f1_macro", "semilla": SEMILLA,
                           "n_entrenamiento": len(ent), "n_prueba": len(prueba), "n_caracteristicas": len(columnas)})

        inicio = time.perf_counter()
        busqueda.fit(X, y, groups=grupos)
        mlflow.log_metric("tiempo_grid_search_s", time.perf_counter() - inicio)

        res = pd.DataFrame(busqueda.cv_results_)
        for i, fila in res.iterrows():                       # una corrida hija por combinación
            with mlflow.start_run(run_name=f"{nombre}-{i:02d}", nested=True):
                mlflow.set_tags({**etiquetas, "modelo": nombre, "etapa": "combinacion"})
                mlflow.log_params({p.split("__")[-1]: fila[f"param_{p}"] for p in rejilla})
                mlflow.log_metrics({**{f"cv_{m}_media": fila[f"mean_test_{m}"] for m in METRICAS_CV},
                                    "cv_f1_macro_std": fila["std_test_f1_macro"],
                                    "entrenamiento_f1_macro_media": fila["mean_train_f1_macro"],
                                    "tiempo_ajuste_s": fila["mean_fit_time"],
                                    "rango_f1_macro": fila["rank_test_f1_macro"]})

        mejor = res.loc[busqueda.best_index_]
        mlflow.log_params({f"mejor_{p.split('__')[-1]}": v for p, v in busqueda.best_params_.items()})
        for k in range(N_FOLDS):
            mlflow.log_metric("cv_f1_macro_fold", mejor[f"split{k}_test_f1_macro"], step=k)
        mlflow.log_metrics({**{f"cv_{m}_media": mejor[f"mean_test_{m}"] for m in METRICAS_CV},
                            "cv_f1_macro_std": mejor["std_test_f1_macro"],
                            "entrenamiento_f1_macro_media": mejor["mean_train_f1_macro"],
                            "brecha_entrenamiento_cv": mejor["mean_train_f1_macro"] - mejor["mean_test_f1_macro"]})

        modelo = busqueda.best_estimator_
        pred = modelo.predict(prueba[columnas])
        met_prueba = metricas(prueba["clase"], pred, "prueba")
        tamano_kb = len(pickle.dumps(modelo)) / 1024                 # costo de desplegarlo en la app
        mlflow.log_metrics({**met_prueba, "tamano_modelo_kb": tamano_kb})

        guardar_figura(figura_matriz(prueba["clase"], pred, f"{nombre}: matriz de confusión en prueba"),
                       f"matriz_confusion_{nombre}.png")
        guardar_figura(figura_rejilla(res, rejilla, nombre), f"rejilla_{nombre}.png")
        mlflow.log_table(res.drop(columns="params").astype({c: str for c in res if c.startswith("param_")}),
                         "resultados/cv_resultados.json")
        mlflow.log_dict(classification_report(prueba["clase"], pred, output_dict=True, zero_division=0),
                        "resultados/reporte_prueba.json")
        # skops rechaza por seguridad los árboles (sus índices de nodo no se validan al predecir); el modelo
        # lo generamos aquí, así que se declara confiable ese único tipo y solo para el random forest
        confiables = ["sklearn.tree._tree.Tree"] if isinstance(modelo, RandomForestClassifier) else None
        info = mlflow.sklearn.log_model(modelo, name="modelo", input_example=X.head(3),
                                        pip_requirements=REQUISITOS_MODELO, skops_trusted_types=confiables)

    print(f"\n{nombre}: mejor {busqueda.best_params_}")
    print(f"  CV f1_macro {mejor['mean_test_f1_macro']:.3f} ± {mejor['std_test_f1_macro']:.3f} | "
          f"prueba f1_macro {met_prueba['prueba_f1_macro']:.3f} | "
          f"enferma como sana {met_prueba['prueba_enferma_como_sana']:.3f}")
    return {"estimador": modelo, "run_id": corrida.info.run_id, "model_uri": info.model_uri,
            "prueba": met_prueba, "tamano_kb": tamano_kb}


def t_corregida(diferencias, proporcion_prueba):
    """Prueba t pareada de Nadeau y Bengio para validación cruzada repetida: (t, p, intervalo de confianza 95 %).

    Los folds comparten fotos de entrenamiento, así que los puntajes no son independientes y la t
    clásica subestima la varianza; la corrección suma n_prueba/n_entrenamiento al factor 1/k.
    El intervalo usa el mismo error estándar corregido, para que sea coherente con la prueba.
    """
    d = np.asarray(diferencias)
    k = len(d)
    error = np.sqrt((1 / k + proporcion_prueba) * d.var(ddof=1))
    t = d.mean() / error
    return t, 2 * stats.t.sf(abs(t), df=k - 1), stats.t.interval(0.95, df=k - 1, loc=d.mean(), scale=error)


def comparar(entrenados, ent, datos, columnas, etiquetas):
    """Validación cruzada repetida con los mejores hiperparámetros, pruebas estadísticas y evaluación por sesión."""
    nombres = list(entrenados)
    X, y, grupos = ent[columnas], ent["clase"], ent["grupo"]

    with mlflow.start_run(run_name="comparacion") as corrida:
        mlflow.set_tags({**etiquetas, "etapa": "comparacion"})
        mlflow.log_params({"repeticiones": REPETICIONES, "n_folds": N_FOLDS, "alfa": ALFA,
                           **{f"run_id_{n}": entrenados[n]["run_id"] for n in nombres}})

        # 1. Mismos folds para los dos modelos (misma semilla por repetición): puntajes pareados
        puntajes = {n: [] for n in nombres}
        for r in range(REPETICIONES):
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA + r)
            for n in nombres:
                puntajes[n].extend(cross_val_score(clone(entrenados[n]["estimador"]), X, y, groups=grupos,
                                                   cv=cv, scoring="f1_macro"))
        a, b = (np.array(puntajes[n]) for n in nombres)
        for paso, (pa, pb) in enumerate(zip(a, b)):
            mlflow.log_metrics({f"rep_f1_macro_{nombres[0]}": pa, f"rep_f1_macro_{nombres[1]}": pb}, step=paso)
        t, p_t, ic = t_corregida(a - b, 1 / (N_FOLDS - 1))
        p_w = stats.wilcoxon(a, b).pvalue
        resumen = {f"rep_f1_macro_media_{n}": np.mean(puntajes[n]) for n in nombres}
        resumen.update({f"rep_f1_macro_std_{n}": np.std(puntajes[n], ddof=1) for n in nombres})
        resumen.update({"diferencia_media": (a - b).mean(), "diferencia_ic95_inf": ic[0],
                        "diferencia_ic95_sup": ic[1], "t_corregida": t, "p_t_corregida": p_t,
                        "p_wilcoxon": p_w, "victorias_" + nombres[0]: int((a > b).sum())})

        # 2. Validación cruzada agrupada por fotos parecidas vs dejando fuera una sesión de captura completa,
        #    con todo el dataset y los mismos hiperparámetros: solo cambia la forma de partir los datos
        sesion_out = {}
        for n in nombres:
            est = entrenados[n]["estimador"]
            pred_grupo = cross_val_predict(clone(est), datos[columnas], datos["clase"], groups=datos["grupo"],
                                           cv=StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEMILLA))
            pred_sesion = cross_val_predict(clone(est), datos[columnas], datos["clase"], groups=datos["sesion"],
                                            cv=LeaveOneGroupOut())
            resumen.update({f"{k}_{n}": v for k, v in metricas(datos["clase"], pred_grupo, "cv_grupos").items()
                            if k.endswith(("f1_macro", "enferma_como_sana"))})
            resumen.update({f"{k}_{n}": v for k, v in metricas(datos["clase"], pred_sesion, "sesion_out").items()})
            sesion_out[n] = pred_sesion
            guardar_figura(figura_matriz(datos["clase"], pred_sesion, f"{n}: sesión fuera"),
                           f"matriz_sesion_out_{n}.png")
        mlflow.log_metrics(resumen)

        # 3. Decisión: si la diferencia es significativa gana el mejor; si no, el más ligero de desplegar
        significativa = p_t < ALFA
        if significativa:
            ganador = nombres[0] if resumen["diferencia_media"] > 0 else nombres[1]
            razon = f"diferencia significativa (p corregida = {p_t:.4f})"
        else:
            ganador = min(nombres, key=lambda n: entrenados[n]["tamano_kb"])
            razon = f"sin diferencia significativa (p corregida = {p_t:.4f}); se elige el modelo más ligero"
        mlflow.set_tags({"ganador": ganador, "razon_decision": razon})

        guardar_figura(figura_comparacion(puntajes), "comparacion_cv_repetida.png")
        guardar_figura(figura_por_sesion(datos, sesion_out), "acierto_por_sesion.png")
        mlflow.log_dict({"puntajes": {n: list(map(float, v)) for n, v in puntajes.items()}},
                        "resultados/puntajes_pareados.json")

    print(f"\nComparación ({REPETICIONES}x{N_FOLDS} folds pareados):")
    for n in nombres:
        print(f"  {n}: {resumen[f'rep_f1_macro_media_{n}']:.3f} ± {resumen[f'rep_f1_macro_std_{n}']:.3f} | "
              f"sesión fuera {resumen[f'sesion_out_f1_macro_{n}']:.3f} | "
              f"CV por grupos (todo el dataset) {resumen[f'cv_grupos_f1_macro_{n}']:.3f}")
    print(f"  diferencia {resumen['diferencia_media']:+.3f} [IC95 corregido {ic[0]:+.3f}, {ic[1]:+.3f}] | "
          f"t corregida p = {p_t:.4f} | Wilcoxon p = {p_w:.4f}")
    print(f"  Ganador: {ganador} ({razon})")
    return ganador, razon, corrida.info.run_id


def figura_comparacion(puntajes):
    """Puntajes pareados: cada línea gris une los dos modelos en el mismo fold."""
    nombres = list(puntajes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for pa, pb in zip(*puntajes.values()):
        ax.plot([0, 1], [pa, pb], color="grey", alpha=0.35, lw=0.8)
    ax.boxplot(list(puntajes.values()), positions=[0, 1], widths=0.35)
    ax.set_xticks([0, 1], nombres)
    ax.set(title=f"F1 macro en {REPETICIONES}x{N_FOLDS} folds pareados", ylabel="F1 macro")
    ax.grid(alpha=0.3, axis="y")
    return fig


def figura_por_sesion(datos, predicciones):
    """Exactitud de cada sesión cuando el modelo no vio ninguna foto de esa sesión."""
    tabla = pd.DataFrame({n: (pred == datos["clase"]).groupby(datos["sesion"]).mean()
                          for n, pred in predicciones.items()})
    tabla["fotos"] = datos.groupby("sesion").size()
    tabla = tabla.sort_values("fotos", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    tabla.drop(columns="fotos").plot.bar(ax=ax, width=0.75)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in tabla["fotos"].items()], rotation=0, fontsize=7)
    ax.set(title="Exactitud por sesión de captura dejada fuera del entrenamiento", ylabel="Exactitud", xlabel="")
    ax.grid(alpha=0.3, axis="y")
    return fig


def registrar_ganador(entrenados, ganador, razon, run_comparacion):
    """Registra el modelo elegido en el Model Registry y le asigna el alias "candidato"."""
    version_reg = mlflow.register_model(entrenados[ganador]["model_uri"], MODELO_REGISTRADO)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(MODELO_REGISTRADO, "candidato", version_reg.version)
    for clave, valor in {"algoritmo": ganador, "razon": razon, "run_comparacion": run_comparacion,
                         **{k: f"{v:.4f}" for k, v in entrenados[ganador]["prueba"].items()}}.items():
        cliente.set_model_version_tag(MODELO_REGISTRADO, version_reg.version, clave, valor)
    print(f"\nModel Registry: {MODELO_REGISTRADO} v{version_reg.version} (alias candidato) <- {ganador}")


def exportar_corridas():
    """Tabla de todas las corridas del experimento en evidencias/, porque mlflow.db no se sube al repo."""
    corridas = mlflow.search_runs(experiment_names=[EXPERIMENTO], order_by=["start_time ASC"])
    columnas = [c for c in corridas.columns
                if c in ("run_id", "start_time", "status") or c.startswith(("tags.mlflow.runName",
                    "tags.mlflow.parentRunId", "tags.modelo",
                    "tags.etapa", "tags.git_commit", "tags.datos_", "tags.ganador", "params.", "metrics."))]
    EVIDENCIAS.mkdir(exist_ok=True)
    corridas[columnas].to_csv(EVIDENCIAS / "corridas_mlflow.csv", index=False)
    return len(corridas)


def entrenar_y_comparar():
    uri = configurar_mlflow(EXPERIMENTO)
    datos, columnas = cargar_datos()
    ent, prueba = separar_prueba(datos)
    etiquetas = {**version_git(),
                 "datos_sha256": huella(DATOS_LIMP / "maiz_parcela.csv", DATOS_LIMP / "caracteristicas.parquet")}
    print(f"Seguimiento: {uri} | experimento: {EXPERIMENTO}")
    print(f"Dataset: {len(datos)} fotos, {len(columnas)} características | "
          f"entrenamiento {len(ent)}, prueba {len(prueba)} (versión de datos {etiquetas['datos_sha256']})")

    entrenados = {nombre: entrenar_modelo(nombre, est, rejilla, n_jobs, ent, prueba, columnas, etiquetas)
                  for nombre, (est, rejilla, n_jobs) in modelos().items()}
    ganador, razon, run_comparacion = comparar(entrenados, ent, datos, columnas, etiquetas)
    registrar_ganador(entrenados, ganador, razon, run_comparacion)
    print(f"Corridas del experimento exportadas: {exportar_corridas()} -> evidencias/corridas_mlflow.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # acentos legibles en la consola de Windows
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validar", action="store_true", help="corrida de prueba de la configuración de MLflow")
    args = parser.parse_args()
    if args.validar:
        raise SystemExit(0 if validar_mlflow() else 1)
    entrenar_y_comparar()
