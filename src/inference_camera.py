# src/inference_camera.py

"""
Script para realizar inferencia en vivo utilizando la cámara del computador
con un modelo YOLOv8 previamente entrenado.

Autor: Daniel Carlosama Martínez
Fecha: 2025-09

Uso:
    uv run src/inference_camera.py

Presiona 'q' para salir de la ventana de video en cualquier momento.
"""

from ultralytics import YOLO
import cv2


def live_inference(model_path):
    """
    Activa la webcam y realiza inferencia en tiempo real usando un modelo YOLOv8.

    Args:
        model_path (str): Ruta al archivo .pt del modelo entrenado.

    Returns:
        None
    """
    print("Activando webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    model = YOLO(model_path)
    print("Modelo cargado:", model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame")
            break

        # Realiza inferencia con el modelo
        results = model.predict(source=frame, stream=True, conf=0.5)

        # Anota los resultados sobre el frame
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Signal For Help - Live", annotated_frame)

        # Presiona 'q' para cerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Finalizando...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_inference("artefactos/best.pt")
