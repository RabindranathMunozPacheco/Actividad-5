"""Funciones de limpieza, estandarización y deduplicación del dataset de hojas de maíz.

Lee datos/datos_ini/manifiesto.csv y las imágenes que descarga fuentes/obtener_datos.py, aplica las
reglas en orden y escribe en datos/datos_limp/:
    registro_limpieza.csv    una fila por imagen del manifiesto con la regla que la excluyó o "conservada"
    maiz_parcela.csv         dataset limpio: una fila por foto con su clase, sesión y grupo de fotos parecidas
    caracteristicas.parquet  1,280 características por foto (MobileNetV2 congelado de la Actividad 4)
    referencia_plantvillage.parquet  las mismas características de la muestra tipo PlantVillage (excluida
                             por R2), solo para comparar en las visualizaciones

Reglas (cada imagen queda con la primera regla que la excluye):
    R1 fuera_de_alcance        carpeta que no es una de las 4 clases del caso (Phaeosphaeria, no identificada)
    R2 tipo_plantvillage       archivo "Corn_*": otra colección mezclada en el repositorio (256x256, casi siempre una
                               hoja aislada en primer plano, muchas sobre fondo gris o negro), no foto de parcela
    R3 etiqueta_dudosa         el sufijo del nombre no corresponde a la carpeta (_SR = roya del sur dentro de CommonRust)
    R4 etiqueta_en_conflicto   la misma imagen (mismo sha256) está en más de una carpeta: tiene varias enfermedades
                               y no hay una sola clase correcta
    R5 duplicado_exacto        mismo sha256 dentro de la misma carpeta: se conserva el primero por nombre
    R6 ilegible                el archivo no se puede abrir como imagen
    R7 conflicto_entre_tomas   toma de la misma hoja (misma sesión, distancia <= DIST_GRUPO) guardada en otra
                               carpeta: es el caso de R4 con archivos distintos, que el sha256 no detecta
    R8 casi_duplicado          hash perceptual a distancia <= DIST_CASI_DUPLICADO de una foto ya conservada
                               (tomas seguidas sin cambio visible): se conserva la primera por nombre
Estandarización: todas las fotos se convierten a RGB y 224x224, y las clases se traducen al español.
Las fotos parecidas sin ser duplicadas (misma sesión y distancia <= DIST_GRUPO: la misma planta con otro
encuadre) se conservan, pero comparten "grupo" para que la validación cruzada no las reparta entre
entrenamiento y prueba. Los umbrales se fijaron revisando pares a cada distancia: hasta 2 la foto es
prácticamente la misma, de 3 a 10 es la misma planta y desde 11 ya son escenas distintas.

Cada ejecución queda registrada en el experimento "maiz-datos" de MLflow (reglas, umbrales, conteos y
archivos), con la etiqueta datos_sha256 que también llevan las corridas de entrenamiento.

Uso:
    python fuentes/datos_prep.py                       # limpieza + características + registro en MLflow
    python fuentes/datos_prep.py --sin-caracteristicas
    python fuentes/datos_prep.py --sin-mlflow
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse.csgraph import connected_components

from obtener_datos import CARPETAS_CASO, MANIFIESTO, RAIZ, REPO, REVISION, carpeta_imagenes, rutas_a_descargar

DATOS_LIMP = RAIZ / "datos" / "datos_limp"
TAMANO = (224, 224)
DIST_CASI_DUPLICADO = 2      # de 64 bits del dHash
DIST_GRUPO = 10
SUFIJO_CARPETA = {"CornLeafBlight": "NCLB", "CommonRust": "CR",
                  "GreyLeafSpot": "GLS", "NoFoliarSymptoms": "NoFoliarSymptoms"}

MODELO_CARACT = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
REVISION_CARACT = "c1861579a670fb6232258805b801cd4137cb7176"


def sufijo(archivo):
    """Etiqueta que trae el nombre del archivo: '..._GLS.jpg' -> 'GLS'."""
    m = re.search(r"_([A-Za-z]+)\.(jpe?g|png)$", archivo, flags=re.IGNORECASE)
    return m.group(1) if m else None


def sesion(archivo):
    """Sesión de captura a partir del nombre: fecha (AAAAMMDD) o prefijo de cámara si no trae fecha."""
    nombre = re.sub(r"^\d+_", "", archivo)               # quita el índice inicial
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", nombre)
    if m:
        return "".join(m.groups())
    m = re.match(r"(20\d{2})_(\d{2})", nombre)            # '2012_03_IMG_6358...' -> '201203'
    if m:
        return "".join(m.groups())
    m = re.search(r"(CIMG|IMG|DSC)", nombre)              # sin fecha: '156_CIMG3892_...' -> 'CIMG'
    return m.group(1) if m else "desconocida"


def dhash(imagen, lado=8):
    """Hash perceptual de diferencias (64 bits): cambia poco si la foto se recomprime o reescala."""
    gris = np.asarray(imagen.convert("L").resize((lado + 1, lado), Image.LANCZOS), dtype=np.int16)
    return (gris[:, 1:] > gris[:, :-1]).flatten()


def reglas_por_metadatos(manifiesto):
    """R1 a R5: se deciden solo con el manifiesto, sin abrir las imágenes."""
    m = manifiesto.copy()
    carpetas_por_sha = m.groupby("sha256")["carpeta"].agg(lambda s: set(s))
    m["regla"] = "conservada"

    def excluir(mascara, regla):
        m.loc[mascara & (m["regla"] == "conservada"), "regla"] = regla

    excluir(~m["carpeta"].isin(CARPETAS_CASO), "R1 fuera_de_alcance")
    excluir(m["archivo"].str.startswith("Corn_"), "R2 tipo_plantvillage")
    excluir(m["archivo"].map(sufijo) != m["carpeta"].map(SUFIJO_CARPETA), "R3 etiqueta_dudosa")
    excluir(m["sha256"].map(carpetas_por_sha).map(len) > 1, "R4 etiqueta_en_conflicto")
    repetido = m.sort_values("archivo").duplicated(["carpeta", "sha256"]).reindex(m.index)
    excluir(repetido, "R5 duplicado_exacto")
    return m


def abrir_estandarizada(ruta):
    """RGB 224x224, o None si el archivo no se puede abrir."""
    try:
        with Image.open(ruta) as im:
            original = (im.mode, im.size)
            im = im.convert("RGB")
            return (im if im.size == TAMANO else im.resize(TAMANO, Image.LANCZOS)), original
    except OSError:
        return None, (None, None)


def reglas_por_imagen(registro, carpeta):
    """R6 a R8, más la estandarización y los grupos de fotos parecidas, sobre las que pasaron R1-R5."""
    registro = registro.assign(modo_original=None, tamano_original=None, relacionada_con=None, grupo=np.nan)
    vivas = registro.index[registro["regla"] == "conservada"]
    hashes, indices = [], []
    for i in vivas:
        imagen, (modo, tamano) = abrir_estandarizada(carpeta / registro.at[i, "ruta"])
        registro.at[i, "modo_original"], registro.at[i, "tamano_original"] = modo, str(tamano)
        if imagen is None:
            registro.at[i, "regla"] = "R6 ilegible"
            continue
        hashes.append(dhash(imagen))
        indices.append(i)

    h = np.array(hashes, dtype=np.uint8)
    dist = (h[:, None, :] != h[None, :, :]).sum(axis=2)          # distancia de Hamming entre todas las parejas
    sesiones = registro.loc[indices, "archivo"].map(sesion).to_numpy()
    clases = registro.loc[indices, "carpeta"].to_numpy()
    misma_toma = (dist <= DIST_GRUPO) & (sesiones[:, None] == sesiones[None, :])

    def marcar(k, regla, j):
        registro.at[indices[k], "regla"] = regla
        registro.at[indices[k], "relacionada_con"] = registro.at[indices[j], "ruta"]

    otra_clase = misma_toma & (clases[:, None] != clases[None, :])
    for k in np.flatnonzero(otra_clase.any(axis=1)):
        marcar(k, "R7 conflicto_entre_tomas", np.flatnonzero(otra_clase[k])[0])

    sin_conflicto = np.flatnonzero(~otra_clase.any(axis=1))
    orden = sin_conflicto[np.argsort(registro.loc[[indices[k] for k in sin_conflicto], "archivo"].to_numpy())]
    conservadas = []
    for k in orden:                                              # se conserva la primera por nombre
        cercana = [j for j in conservadas if dist[k, j] <= DIST_CASI_DUPLICADO]
        if cercana:
            marcar(k, "R8 casi_duplicado", cercana[0])
        else:
            conservadas.append(k)

    conservadas = np.array(sorted(conservadas))
    _, etiquetas = connected_components(misma_toma[np.ix_(conservadas, conservadas)].astype(np.int8), directed=False)
    for k, g in zip(conservadas, etiquetas):
        registro.at[indices[k], "grupo"] = int(g)
    return registro


def caracteristicas(limpio, carpeta, lote=32):
    """Vector de 1,280 características por foto con el MobileNetV2 de la Actividad 4, sin su capa final."""
    import torch
    # Procesador con backend PIL explícito: MobileNetV2ImageProcessor elige backend según si torchvision está
    # instalado, y cada uno redimensiona distinto, así que las características cambiarían de un equipo a otro
    from transformers import MobileNetV2ImageProcessorPil, MobileNetV2Model
    procesador = MobileNetV2ImageProcessorPil.from_pretrained(MODELO_CARACT, revision=REVISION_CARACT)
    modelo = MobileNetV2Model.from_pretrained(MODELO_CARACT, revision=REVISION_CARACT).eval()
    vectores = []
    for inicio in range(0, len(limpio), lote):
        imagenes = [abrir_estandarizada(carpeta / r)[0] for r in limpio["ruta"].iloc[inicio:inicio + lote]]
        with torch.no_grad():
            vectores.append(modelo(**procesador(imagenes, return_tensors="pt")).pooler_output.numpy())
    tabla = pd.DataFrame(np.vstack(vectores).astype(np.float32),
                         columns=[f"f{i:04d}" for i in range(vectores[0].shape[1])])
    tabla.insert(0, "id", limpio["id"].to_numpy())
    return tabla


def preparar(con_caracteristicas=True):
    manifiesto = pd.read_csv(MANIFIESTO)
    carpeta = carpeta_imagenes(rutas_a_descargar(manifiesto))
    registro = reglas_por_imagen(reglas_por_metadatos(manifiesto), carpeta)

    limpio = registro[registro["regla"] == "conservada"].sort_values("ruta", ignore_index=True)
    limpio = pd.DataFrame({
        "id": [f"m{i:04d}" for i in range(len(limpio))],
        "ruta": limpio["ruta"], "archivo": limpio["archivo"], "sha256": limpio["sha256"],
        "etiqueta_original": limpio["carpeta"], "clase": limpio["carpeta"].map(CARPETAS_CASO),
        "sesion": limpio["archivo"].map(sesion), "grupo": limpio["grupo"].astype(int),
        "modo_original": limpio["modo_original"], "tamano_original": limpio["tamano_original"],
    })

    DATOS_LIMP.mkdir(parents=True, exist_ok=True)
    registro.to_csv(DATOS_LIMP / "registro_limpieza.csv", index=False)
    limpio.to_csv(DATOS_LIMP / "maiz_parcela.csv", index=False)
    if con_caracteristicas:
        caracteristicas(limpio, carpeta).to_parquet(DATOS_LIMP / "caracteristicas.parquet", index=False)
        # Muestra tipo PlantVillage excluida por R2: no entrena, solo sirve de referencia en la visualización
        descargadas = set(rutas_a_descargar(manifiesto))
        muestra = registro[(registro["regla"] == "R2 tipo_plantvillage") & registro["ruta"].isin(descargadas)]
        muestra = pd.DataFrame({"id": [f"pv{i:03d}" for i in range(len(muestra))], "ruta": muestra["ruta"].to_numpy(),
                                "clase": muestra["carpeta"].map(CARPETAS_CASO).to_numpy()})
        referencia = caracteristicas(muestra, carpeta)
        referencia.insert(1, "clase", muestra["clase"])
        referencia.to_parquet(DATOS_LIMP / "referencia_plantvillage.parquet", index=False)
    return registro, limpio


def registrar_en_mlflow(registro, limpio):
    """Corrida "preparacion" en el experimento maiz-datos: origen, umbrales, conteo por regla y archivos."""
    from train import CLASES, configurar_mlflow, huella, version_git   # antes que mlflow: train silencia sus avisos
    import mlflow
    configurar_mlflow("maiz-datos")
    huella_datos = huella(DATOS_LIMP / "maiz_parcela.csv", DATOS_LIMP / "caracteristicas.parquet")
    with mlflow.start_run(run_name="preparacion") as corrida:
        mlflow.set_tags({**version_git(), "etapa": "datos", "datos_sha256": huella_datos})
        mlflow.log_params({"repo_origen": REPO, "revision_origen": REVISION,
                           "modelo_caracteristicas": MODELO_CARACT, "revision_caracteristicas": REVISION_CARACT,
                           "tamano": f"{TAMANO[0]}x{TAMANO[1]}", "dist_casi_duplicado": DIST_CASI_DUPLICADO,
                           "dist_grupo": DIST_GRUPO})
        conteo = registro["regla"].value_counts()
        mlflow.log_metrics({**{("conservadas" if r == "conservada" else "excluidas_" + r.replace(" ", "_").lower()): int(n)
                               for r, n in conteo.items()},
                            **{f"clase_{CLASES[c]}": int(n) for c, n in limpio["clase"].value_counts().items()},
                            "imagenes_origen": len(registro), "grupos": limpio["grupo"].nunique(),
                            "sesiones": limpio["sesion"].nunique()})
        mlflow.log_input(mlflow.data.from_pandas(limpio, source=str(DATOS_LIMP), name="maiz_parcela",
                                                 targets="clase"), context="limpio")
        for archivo in (MANIFIESTO, DATOS_LIMP / "registro_limpieza.csv", DATOS_LIMP / "maiz_parcela.csv"):
            mlflow.log_artifact(str(archivo), "datos")
    return corrida.info.run_id, huella_datos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sin-caracteristicas", action="store_true", help="no calcula caracteristicas.parquet")
    parser.add_argument("--sin-mlflow", action="store_true", help="no registra la corrida en MLflow")
    args = parser.parse_args()

    registro, limpio = preparar(con_caracteristicas=not args.sin_caracteristicas)
    print("Imágenes por regla:")
    print(registro["regla"].value_counts().sort_index().to_string())
    print(f"\nDataset limpio: {len(limpio)} fotos, {limpio['grupo'].nunique()} grupos de fotos parecidas")
    print(limpio["clase"].value_counts().to_string())
    if not args.sin_mlflow:
        run_id, huella_datos = registrar_en_mlflow(registro, limpio)
        print(f"\nMLflow: experimento maiz-datos, corrida {run_id} (datos_sha256 {huella_datos})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
