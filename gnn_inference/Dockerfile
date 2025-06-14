FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# 1. Install PyTorch first
RUN pip install torch==2.2.2

# 2. Install the rest (torch-geometric, torch-scatter, torch-sparse, fastapi, uvicorn, etc)
RUN pip install -r requirements.txt --no-deps --verbose

COPY . .

EXPOSE 10000

CMD ["uvicorn", "model_serve_app:app", "--host", "0.0.0.0", "--port", "10000"]
