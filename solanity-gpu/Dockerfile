# Stage 1: Build the CUDA application
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# Install build tools
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . /app

# Build the CUDA application
# -B forces rebuild
RUN make -B

# Generate precomputed tables during build
# This requires running the binary, so we need the runtime libs present in this image (which they are)
RUN ./solanity --generate-tables

# Stage 2: Runtime environment (Python 3.10 Slim as requested)
FROM python:3.10-slim

WORKDIR /app

# Copy artifacts from builder
COPY --from=builder /app/solanity /app/solanity
COPY --from=builder /app/precomp_tables.bin /app/precomp_tables.bin
COPY --from=builder /app/worker.py /app/worker.py
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Attempt to copy CUDA runtime libraries from builder to runtime
# The builder is Ubuntu (glibc), python:3.10-slim is Debian (glibc).
# We copy libcudart.so.11.0 and symlinks.
# Location in nvidia image is typically /usr/local/cuda/lib64
COPY --from=builder /usr/local/cuda/lib64/libcudart.so* /usr/lib/x86_64-linux-gnu/

# Install Python requirements
# We use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Set entry point
ENTRYPOINT ["python", "worker.py"]
