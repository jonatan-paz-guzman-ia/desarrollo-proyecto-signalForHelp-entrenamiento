# Imagen base
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Evitar archivos .pyc y usar stdout para logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (aprovecha la cache)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Comando por defecto (puedes sobreescribirlo en docker run)
CMD ["python", "src/train.py", "--data", "data/dataset.yaml", "--epochs", "2", "--img", "640"]
