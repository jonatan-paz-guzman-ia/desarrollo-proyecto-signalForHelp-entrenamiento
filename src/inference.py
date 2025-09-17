# src/inference.py

"""
Script para realizar inferencia sobre imágenes o carpetas de imágenes
utilizando un modelo YOLOv8 previamente entrenado.

Fecha: 2025-09
Uso:
    uv run src/inference.py --source data/test/images
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def predict(model_path, source, save_dir="artefactos/predicciones"):
    """
    Realiza inferencia utilizando un modelo YOLOv8 y guarda los resultados en disco.

    Args:
        model_path (str): Ruta al archivo de pesos .pt del modelo YOLOv8.
        source (str): Ruta a la imagen o carpeta de imágenes para inferencia.
        save_dir (str, opcional): Directorio donde se guardarán los resultados. Por defecto es 'artefactos/predicciones'.

    Returns:
        None
    """
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=save_dir,
        name="pred",
        exist_ok=True
    )
    print(f"Inferencia completada. Resultados guardados en: {save_dir}/pred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realizar inferencia con modelo YOLOv8 entrenado"
    )
    parser.add_argument(
        "--model", type=str, default="artefactos/best.pt",
        help="Ruta del modelo .pt entrenado"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Ruta de la imagen o carpeta de prueba"
    )

    args = parser.parse_args()
    predict(args.model, args.source)
