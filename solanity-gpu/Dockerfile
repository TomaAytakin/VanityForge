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
RUN make

# Set entry point
ENTRYPOINT ["./solanity"]
