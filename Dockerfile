
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU versions of PyTorch and torchvision
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1" "torchvision>=0.16"

# Copy project files
COPY . .

# Install project and development dependencies
RUN pip install --no-cache-dir ".[dev]"

# Run classifier CLI
ENTRYPOINT ["wheat-classifier"]
