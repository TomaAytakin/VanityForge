FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential git pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install the tool directly.
# It will be compiled and placed in /root/.cargo/bin/solana-vanity
RUN cargo install solana-vanity

# We do NOT need 'cargo build --release' because 'install' already did it.

# Copy the worker script
COPY worker.py .

# IMPORTANT: The binary is named 'solana-vanity', not 'vanity'
CMD ["python3", "worker.py"]
