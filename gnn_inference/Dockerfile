FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code and model files into image
COPY . .

EXPOSE 10000

# Start server
CMD ["uvicorn", "model_serve_app:app", "--host", "0.0.0.0", "--port", "10000"]
