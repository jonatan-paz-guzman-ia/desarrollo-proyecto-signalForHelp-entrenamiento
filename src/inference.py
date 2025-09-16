# src/inference.py

from ultralytics import YOLO
import argparse
from pathlib import Path

def predict(model_path, source, save_dir="artefactos/predicciones"):
    model = YOLO(model_path)
    results = model.predict(source=source, save=True, project=save_dir, name="pred", exist_ok=True)
    print(f"Inferencia completada. Resultados guardados en: {save_dir}/pred")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar inferencia con modelo YOLOv8 entrenado")
    parser.add_argument("--model", type=str, default="artefactos/best.pt", help="Ruta del modelo .pt")
    parser.add_argument("--source", type=str, required=True, help="Ruta de la imagen o carpeta de prueba")

    args = parser.parse_args()
    predict(args.model, args.source)
