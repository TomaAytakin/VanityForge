FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential git pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install the high-performance Solana vanity tool
# We use 'solana-vanity' crate which has GPU optimizations or a similar proven fork
# cloning the 'gpu' branch of a known high-perf repo
RUN git clone https://github.com/lmc-labs/solana-vanity-gen-gpu.git . || cargo install solana-vanity

# Build with CUDA optimizations
RUN cargo build --release

# Copy the worker script
COPY worker.py .

CMD ["python3", "worker.py"]
