FROM pathwaycom/pathway:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers numpy

WORKDIR /app