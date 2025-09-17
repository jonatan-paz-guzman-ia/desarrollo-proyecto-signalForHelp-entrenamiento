# src/train.py

"""
Script para entrenar un modelo YOLOv8 para segmentación de gestos de auxilio ("Signal for Help").

Este archivo permite configurar las rutas del dataset, el número de épocas,
el tamaño de las imágenes y el tipo de modelo a utilizar.

Autor: Daniel Carlosama Martínez
Fecha: 2025-09

Uso:
    uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def train_model(data_yaml, epochs, img_size, model_type, save_dir):
    """
    Entrena un modelo YOLOv8 para segmentación de imágenes.

    Args:
        data_yaml (str): Ruta al archivo YAML del dataset con estructura Roboflow.
        epochs (int): Número de épocas de entrenamiento.
        img_size (int): Dimensión de las imágenes (cuadradas).
        model_type (str): Tipo de modelo YOLOv8 (ej. 'yolov8n-seg.pt').
        save_dir (str): Carpeta donde se guardarán los resultados.

    Returns:
        None
    """
    print(f"Entrenando modelo: {model_type}")
    print(f"Dataset: {data_yaml}")
    print(f"Épocas: {epochs}, Tamaño de imagen: {img_size}")

    model = YOLO(model_type)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        save=True,
        project=save_dir,
        name="signalforhelp",
        exist_ok=True
    )

    # Copia el modelo entrenado final (best.pt) a la carpeta artefactos
    best_model = Path(save_dir) / "signalforhelp" / "weights" / "best.pt"
    target = Path("artefactos") / "best.pt"
    if best_model.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        best_model.replace(target)
        print(f"✅ Modelo guardado en: {target.resolve()}")
    else:
        print("⚠️ No se encontró el modelo entrenado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8 Signal for Help")
    parser.add_argument("--data", type=str, required=True, help="Ruta al archivo YAML del dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--img", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Modelo base YOLOv8")
    parser.add_argument("--save-dir", type=str, default="runs/train", help="Directorio de salida")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img,
        model_type=args.model,
        save_dir=args.save_dir
    )
