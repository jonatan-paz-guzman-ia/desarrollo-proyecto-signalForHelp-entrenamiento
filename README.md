# 🧠 Proyecto: Signal For Help - Entrenamiento de modelo YOLOv8

Este repositorio entrena un modelo de segmentación con **YOLOv8n-seg** para identificar gestos de auxilio (✋ palma abierta, ✊ puño cerrado), como parte de una solución de visión computacional en contextos de emergencia.

---

## ⚙️ Requisitos del entorno

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gestor de entorno y dependencias)
- Git
- `ffmpeg`, `libgl1` (solo para sistemas Linux o Docker)

---

## 🔧 Instalación

```bash
# Clona el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Crea entorno virtual e instala dependencias
uv venv
uv pip install -r requirements.txt


```
## 📈 Seguimiento con MLflow

```bash
mlflow ui
```

## 🚀 Entrenamiento del modelo

```bash
uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640
```
Abre tu navegador en: http://127.0.0.1:5000

Aquí puedes:

Ver parámetros de entrenamiento
Visualizar métricas por epoch
Descargar el modelo entrenado

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