# tests/test_inference.py

"""
Tests unitarios para validación básica de inferencia con YOLOv8.

Asegura que el modelo se carga correctamente y genera resultados válidos
sobre una imagen de prueba.
"""

import os
from ultralytics import YOLO

# Ruta al modelo entrenado
MODEL_PATH = "artefactos/best.pt"

# Ruta a una imagen de prueba
TEST_IMAGE = "data/test/images/puno_3_jpg.rf.71251e31c7a6a97c27b64a524f3c54cc.jpg"

def test_model_loads():
    """Test: El modelo se carga correctamente desde archivo."""
    model = YOLO(MODEL_PATH)
    assert model is not None

def test_predict_image_runs():
    """Test: Se puede realizar inferencia sin errores sobre una imagen."""
    model = YOLO(MODEL_PATH)
    results = model.predict(source=TEST_IMAGE)
    assert results is not None
    assert len(results) > 0

def test_prediction_outputs_are_valid():
    """Test: Las predicciones tienen atributos esperados."""
    model = YOLO(MODEL_PATH)
    results = model.predict(source=TEST_IMAGE)
    for r in results:
        assert hasattr(r, 'boxes')
        assert hasattr(r, 'masks')
