# src/train.py

"""
Script para entrenar un modelo YOLOv8 para segmentación de gestos de auxilio ("Signal for Help").

Este archivo permite configurar las rutas del dataset, el número de épocas,
el tamaño de las imágenes y el tipo de modelo a utilizar. Ahora incluye MLflow para tracking de experimentos.

Autor: Daniel Carlosama Martínez
Fecha: 2025-09

Uso:
    uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import mlflow
import mlflow.pyfunc


def train_model(data_yaml, epochs, img_size, model_type, save_dir):
    """
    Entrena un modelo YOLOv8 para segmentación de imágenes y registra el experimento con MLflow.

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

    # Inicia el experimento en MLflow
    mlflow.set_experiment("SignalForHelp - YOLOv8")
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("dataset", data_yaml)

        # Entrena el modelo
        model = YOLO(model_type)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            save=True,
            project=save_dir,
            name="signalforhelp",
            exist_ok=True
        )

        # Ruta al mejor modelo
        best_model = Path(save_dir) / "signalforhelp" / "weights" / "best.pt"
        target = Path("artefactos") / "best.pt"
        if best_model.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            best_model.replace(target)
            print(f"✅ Modelo guardado en: {target.resolve()}")

            # Registra el modelo en MLflow
            mlflow.log_artifact(str(target), artifact_path="model")
        else:
            print("⚠️ No se encontró el modelo entrenado.")

        # Registrar algunas métricas básicas si están disponibles
        try:
            metrics = results.metrics
            mlflow.log_metric("precision", metrics.box.map if metrics.box else 0)
            mlflow.log_metric("recall", metrics.box.map50 if metrics.box else 0)
            mlflow.log_metric("segmentation_mAP", metrics.seg.map if metrics.seg else 0)
        except Exception as e:
            print("No se pudieron registrar métricas en MLflow:", e)


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
