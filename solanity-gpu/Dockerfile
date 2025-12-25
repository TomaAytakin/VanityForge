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

# Stage 2: Runtime environment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and curl
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts from builder
COPY --from=builder /app/solanity /app/solanity
COPY --from=builder /app/precomp_tables.bin /app/precomp_tables.bin
COPY --from=builder /app/worker.py /app/worker.py
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Make binary executable
RUN chmod +x /app/solanity

# Install Python requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Set entry point
CMD ["python3", "worker.py"]
