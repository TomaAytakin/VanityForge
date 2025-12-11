FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Install build tools and Python
# Ubuntu 20.04 might need python3-pip explicitly.
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
RUN make

# Install Python requirements
RUN pip3 install -r requirements.txt

# Set entry point to Python wrapper
ENTRYPOINT ["python3", "worker.py"]
