from ultralytics import YOLO
import cv2

def live_inference(model_path):
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

        # Realiza inferencia (NO muestra automáticamente)
        results = model.predict(source=frame, stream=True, conf=0.5)

        # Itera sobre resultados y dibuja manualmente
        for r in results:
            annotated_frame = r.plot()  # Dibuja sobre el frame

            # Muestra en pantalla el frame anotado
            cv2.imshow("Signal For Help - Live", annotated_frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Finalizando...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_inference("artefactos/best.pt")
