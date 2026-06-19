# === Stage 1: Builder — install deps & pre-cache model ===
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.5.0,<2.6.0 \
    && pip install --no-cache-dir -r requirements.txt

# Pre-cache the sentence transformer model so it's baked into the image
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# === Stage 2: Runtime — minimal production image ===
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy huggingface model cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY src/ src/

# Create a non-root user for security
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --gid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
