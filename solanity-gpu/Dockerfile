# Stage 1: Build the CUDA application
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build binary
RUN make -B

# Generate precomputed tables during build
RUN ./solanity --generate-tables

# Stage 2: Runtime environment
# We recommend using the NVIDIA CUDA Runtime image to ensure all libraries are present.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python runtime
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

# Copy entrypoint script
COPY --from=builder /app/entrypoint.sh .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app

# Set permissions
RUN chmod +x solanity entrypoint.sh

# Switch to non-root user
USER appuser

# Set entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "worker.py"]
