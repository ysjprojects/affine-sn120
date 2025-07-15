FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install python deps (wheel cache in layer)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Expose nothing – validator runs as a cron-style process
CMD ["python", "-m", "affine", "validate", "--delay", "5"] 