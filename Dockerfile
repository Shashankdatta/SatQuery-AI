FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

WORKDIR /app

# Install system libraries needed for imaging and computer vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install lightweight CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python backend dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app code
COPY backend/app /app/app

# Expose port (Render injects $PORT, default is 10000)
EXPOSE 10000 8000 7860

# Launch Uvicorn with dynamic port binding
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
