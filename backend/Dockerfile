FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install heavy PyTorch and Machine Learning libraries 
# Using specific trusted versions and the cpu wheels where applicable to save space (since this is generic docker)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    transformers \
    pillow \
    numpy \
    opencv-python-headless \
    ultralytics \
    onnx \
    onnxruntime \
    websockets

# Create required directory structures to prevent startup crashes
RUN mkdir -p /app/backend /app/models /app/data/active_learning_review

# Copy the backend source code
COPY . /app/backend/

# Expose backend API port
EXPOSE 8000

# Run the FastAPI server mapped globally
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
