# datos_limp

Datos después de la limpieza, estandarización y deduplicación aplicadas por `fuentes/datos_prep.py`. Son la entrada de `fuentes/train.py` y de `fuentes/entrena.ipynb`.

| Archivo | Contenido |
|---|---|
| `registro_limpieza.csv` | las 7,012 imágenes del manifiesto con la regla que las excluyó (o `conservada`), la foto relacionada en R7/R8 y el grupo |
| `maiz_parcela.csv` | dataset limpio: 1,078 fotos de parcela con clase, sesión de captura y grupo de fotos parecidas |
| `caracteristicas.parquet` | 1,280 características por foto (MobileNetV2 congelado), unidas por `id` |
| `referencia_plantvillage.parquet` | las mismas características de 400 imágenes `Corn_*` excluidas por R2; solo para visualización |

Diccionario de datos y criterio de cada regla: README principal, sección 3. Los archivos se regeneran idénticos byte por byte con `python fuentes/datos_prep.py`.
