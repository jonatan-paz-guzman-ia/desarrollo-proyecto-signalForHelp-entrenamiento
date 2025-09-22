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


def log_all_metrics(results):
    """Registra todas las métricas relevantes de YOLOv8 en MLflow."""
    try:
        # Métricas de cajas (detección)
        if hasattr(results, "box") and results.box:
            mlflow.log_metric("box_precision", results.box.mp)
            mlflow.log_metric("box_recall", results.box.mr)
            mlflow.log_metric("box_map50", results.box.map50)
            mlflow.log_metric("box_map", results.box.map)

        # Métricas de segmentación
        if hasattr(results, "seg") and results.seg:
            mlflow.log_metric("seg_precision", results.seg.mp)
            mlflow.log_metric("seg_recall", results.seg.mr)
            mlflow.log_metric("seg_map50", results.seg.map50)
            mlflow.log_metric("seg_map", results.seg.map)

        # Velocidades (inferencia, NMS, etc.)
        if hasattr(results, "speed") and isinstance(results.speed, dict):
            for k, v in results.speed.items():
                mlflow.log_metric(f"speed_{k}", v)

        print("✅ Métricas registradas en MLflow")
    except Exception as e:
        print("⚠️ No se pudieron registrar métricas en MLflow:", e)


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

    # Registro manual en MLflow
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

        # Métricas completas
        log_all_metrics(results)


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
