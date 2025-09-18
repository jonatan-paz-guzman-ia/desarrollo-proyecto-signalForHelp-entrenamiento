# Dockerfile

# Imagen base con Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY . .

# Instalar sistema y dependencias del proyecto
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV (gestor de entorno)
RUN pip install uv

# Crear entorno virtual y activar
RUN uv venv && \
    uv pip install -r requirements.txt

# Comando por defecto (puedes cambiar por camera o train si prefieres)
CMD ["uv", "run", "src/inference.py", "--source", "data/test/images"]
