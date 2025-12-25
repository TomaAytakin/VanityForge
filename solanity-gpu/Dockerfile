# Stage 1: Build the CUDA application
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install build dependencies
# We need python3 in builder to potentially run scripts or if the build uses it
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build binary
# The Makefile uses nvcc
RUN make -B

# Generate precomputed tables during build
# This requires running the binary, so we need the runtime libs present in this image (which they are in devel)
RUN ./solanity --generate-tables

# Stage 2: Runtime environment
# Use the official NVIDIA CUDA Runtime image to ensure all shared libraries are present
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python runtime and curl (useful for healthchecks/debugging)
# Using --no-install-recommends to keep image size small
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts from builder
COPY --from=builder /app/solanity .
COPY --from=builder /app/precomp_tables.bin .
COPY --from=builder /app/worker.py .
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user for security (Cloud Run best practice)
RUN useradd -m appuser && chown -R appuser:appuser /app

# Set executable permissions (must be done before switching user or by root)
RUN chmod +x solanity

# Switch to non-root user
USER appuser

# Set entry point
CMD ["python3", "worker.py"]
