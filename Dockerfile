# CPU image for the API and the analysis pipeline.
# Build:  docker build -t concall .
# Run:    docker run -p 8000:8000 --env-file .env -v %cd%/data:/app/data -v %cd%/models:/app/models concall
# (data/ and models/ are mounted, not baked in, so the image stays small and results persist.)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch keeps the image ~2 GB smaller than the CUDA build.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -e .

ENV HF_HOME=/app/models
EXPOSE 8000
CMD ["uvicorn", "concall.api:app", "--host", "0.0.0.0", "--port", "8000"]
