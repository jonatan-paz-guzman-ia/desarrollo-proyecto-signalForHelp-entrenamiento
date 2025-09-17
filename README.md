# 🧠 Proyecto: Signal For Help - Entrenamiento de modelo YOLOv8

Este proyecto entrena un modelo de segmentación con YOLOv8 para identificar gestos de ayuda (palma abierta y puño cerrado), como parte de una solución de visión computacional en contextos de emergencia.

## ⚙️ Requisitos

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (gestor de entornos recomendado)
- Git
- ffmpeg, libgl1 (solo para entorno Docker o sistemas Linux)

## 🔧 Instalación

```bash
uv venv
uv pip install -r requirements.txt

```
## 🚀 Entrenamiento del modelo

```bash
uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640
```

## 📸 Inferencia por imagen

```bash
uv run src/inference.py --source data/test/images
```

## 🎥 Inferencia con cámara en vivo

```bash
uv run src/inference_camera.py
```

## 🧪 Pruebas unitarias

```bash
uv run pytest tests/
```

Incluye:
    - Carga del modelo
    - Ejecución de inferencia
    - Validación de resultados

## 🐳 Docker (opcional)

```bash
docker build -t signalforhelp-inference .
docker run --rm signalforhelp-inference
```
## 📁 Estructura del proyecto

```bash
├── src/
│   ├── train.py
│   ├── inference.py
│   └── inference_camera.py
├── tests/
│   └── test_inference.py
├── artefactos/
│   └── best.pt
├── data/
│   └── dataset.yaml
│   └── test/train/valid/
├── Dockerfile
├── requirements.txt
├── README.md
```
## 👤 Autores

Jonatan Paz Guzman
GitHub: @jonatan-paz-guzman-ia

Dayana Muñoz Muñoz
GitHub: @dayana-IA

Daniel Carlosama Martínez
GitHub: @21danka