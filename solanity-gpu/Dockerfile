FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install build tools and Python
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . /app

# Build the CUDA application
RUN make -B

# Generate precomputed tables during build to avoid startup lag
RUN ./solanity --generate-tables

# Install Python requirements
RUN pip3 install -r requirements.txt

# Set entry point to Python wrapper
ENTRYPOINT ["python3", "worker.py"]
