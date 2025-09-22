# src/train.py

"""
Script para entrenar un modelo YOLOv8 para segmentación de gestos de auxilio ("Signal for Help").

Ahora:
- Entrena el modelo con Ultralytics.
- Desactiva cualquier callback automático de MLflow en Ultralytics.
- Después del entrenamiento, registra parámetros, artefactos y métricas en MLflow de forma manual.

Fecha: 2025-09

Uso:
    uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import mlflow


def train_model(data_yaml, epochs, img_size, model_type, save_dir):
    print(f"Entrenando modelo: {model_type}")
    print(f"Dataset: {data_yaml}")
    print(f"Épocas: {epochs}, Tamaño de imagen: {img_size}")

    # Cargar modelo
    model = YOLO(model_type)

    # Eliminar callback automático de MLflow de Ultralytics
    if "mlflow" in model.callbacks:
        del model.callbacks["mlflow"]

    # Entrenar modelo
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        save=True,
        project=save_dir,
        name="signalforhelp",
        exist_ok=True,
    )

    # Guardar best.pt en artefactos
    best_model = Path(save_dir) / "signalforhelp" / "weights" / "best.pt"
    target = Path("artefactos") / "best.pt"
    if best_model.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        best_model.replace(target)
        print(f"✅ Modelo guardado en: {target.resolve()}")
    else:
        print("⚠️ No se encontró el modelo entrenado.")

    # Registro manual en MLflow (después del entrenamiento)
    mlflow.set_experiment("SignalForHelp - YOLOv8")
    with mlflow.start_run():
        # Parámetros
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("dataset", data_yaml)

        # Artefacto del modelo
        if target.exists():
            mlflow.log_artifact(str(target), artifact_path="model")

        # Métricas básicas
        try:
            metrics = results.metrics
            mlflow.log_metric("precision", getattr(metrics.box, "map", 0))
            mlflow.log_metric("recall", getattr(metrics.box, "map50", 0))
            mlflow.log_metric("segmentation_mAP", getattr(metrics.seg, "map", 0))
        except Exception as e:
            print("⚠️ No se pudieron registrar métricas en MLflow:", e)


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
