FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install System Deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Update Path
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Logic
RUN cargo install solana-vanity

# Setup Python
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Setup Worker
COPY worker.py .

# Command
CMD ["python3", "worker.py"]
