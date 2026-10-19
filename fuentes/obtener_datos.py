"""Script de obtención del dataset: fotos de hojas de maíz del repositorio de campo de Hugging Face.

1. Lista todas las imágenes del repositorio en un commit fijo y guarda el manifiesto
   datos/datos_ini/manifiesto.csv (ruta, carpeta, archivo, bytes, sha256).
2. Descarga las imágenes que usa la limpieza: todas las fotos de parcela de las 4 clases del caso
   y una muestra de las imágenes tipo PlantVillage (solo para la visualización de anomalías).
3. Verifica que el sha256 de cada archivo descargado coincide con el del manifiesto.

Las imágenes no se suben al repositorio: quedan en la caché de Hugging Face (~/.cache/huggingface)
y el manifiesto con su sha256 permite comprobar que son exactamente las mismas en cualquier equipo.
No necesita token (el dataset es público).

Uso:
    python fuentes/obtener_datos.py                    # manifiesto + descarga + verificación
    python fuentes/obtener_datos.py --solo-manifiesto
"""
import argparse
import hashlib
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Windows sin modo desarrollador no admite symlinks: la caché copia los archivos y avisa en cada corrida
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

REPO = "muAtarist/maize_foliar_diseases"
REVISION = "448bec6c7abe18fe87917e443368c63ed3786119"   # commit fijo: el dataset no cambia aunque el autor lo actualice
RAIZ = Path(__file__).resolve().parents[1]
MANIFIESTO = RAIZ / "datos" / "datos_ini" / "manifiesto.csv"
SEMILLA = 42

# Carpetas del repositorio que corresponden a las 4 clases del caso de uso
CARPETAS_CASO = {"CornLeafBlight": "Tizón", "CommonRust": "Roya común",
                 "GreyLeafSpot": "Mancha gris", "NoFoliarSymptoms": "Sana"}
MUESTRA_PLANTVILLAGE = 100   # por clase


def crear_manifiesto():
    """Una fila por imagen del repositorio, tal como viene en el origen."""
    archivos = HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, recursive=True)
    filas = [{"ruta": f.path, "carpeta": f.path.split("/")[1], "archivo": f.path.split("/")[-1],
              "bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
             for f in archivos if f.path.lower().endswith((".jpg", ".jpeg", ".png"))]
    manifiesto = pd.DataFrame(filas).sort_values("ruta", ignore_index=True)
    MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    manifiesto.to_csv(MANIFIESTO, index=False)
    return manifiesto


def rutas_a_descargar(manifiesto):
    """Fotos de parcela de las 4 clases + muestra de las tipo PlantVillage (archivos "Corn_*")."""
    caso = manifiesto[manifiesto["carpeta"].isin(CARPETAS_CASO)]
    plantvillage = caso["archivo"].str.startswith("Corn_")
    rng = random.Random(SEMILLA)
    muestra = [r for _, g in caso[plantvillage].groupby("carpeta")
               for r in rng.sample(sorted(g["ruta"]), MUESTRA_PLANTVILLAGE)]
    return sorted(caso.loc[~plantvillage, "ruta"]) + sorted(muestra)


def carpeta_imagenes(rutas):
    """Descarga (o reutiliza de la caché) las imágenes y devuelve la carpeta local del snapshot.

    Se pide archivo por archivo: con miles de rutas exactas, el filtro allow_patterns de snapshot_download
    compara cada archivo del repositorio contra cada patrón y tarda varios minutos solo en filtrar.
    Con la revisión fija, un archivo que ya está en la caché se devuelve sin consultar la red.
    """
    disable_progress_bars()
    with ThreadPoolExecutor(max_workers=16) as pool:
        locales = list(pool.map(lambda r: hf_hub_download(REPO, r, repo_type="dataset", revision=REVISION), rutas))
    return Path(locales[0]).parents[len(Path(rutas[0]).parts) - 1]


def verificar(manifiesto, carpeta, rutas):
    """Archivos cuyo sha256 local no coincide con el del manifiesto (debería ser una lista vacía)."""
    esperado = manifiesto.set_index("ruta")["sha256"]
    return [r for r in rutas
            if hashlib.sha256((carpeta / r).read_bytes()).hexdigest() != esperado[r]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-manifiesto", action="store_true", help="no descarga imágenes")
    args = parser.parse_args()

    manifiesto = crear_manifiesto()
    print(f"Manifiesto: {MANIFIESTO.relative_to(RAIZ)} ({len(manifiesto)} imágenes)")
    print(manifiesto["carpeta"].value_counts().to_string())
    if args.solo_manifiesto:
        return

    rutas = rutas_a_descargar(manifiesto)
    carpeta = carpeta_imagenes(rutas)
    malos = verificar(manifiesto, carpeta, rutas)
    print(f"\nDescargadas {len(rutas)} imágenes en {carpeta}")
    print(f"Verificación sha256: {'OK' if not malos else f'{len(malos)} archivos no coinciden'}")
    if malos:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
