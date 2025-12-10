FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . /app

# Build the CUDA application
# Using generic architecture flags compatible with L4 (Ada Lovelace, sm_89) and others
RUN nvcc -o solanity main.cu -O3 --generate-code arch=compute_89,code=sm_89 --generate-code arch=compute_75,code=sm_75 --generate-code arch=compute_80,code=sm_80

# Set entry point
ENTRYPOINT ["./solanity"]
