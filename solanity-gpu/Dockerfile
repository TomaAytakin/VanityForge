FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Prevent "Select Timezone" prompt hanging the build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python, AND Pip (safety net)
RUN apt-get update && apt-get install -y \
    curl build-essential git pkg-config libssl-dev \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install the engine (Pre-compiled)
RUN cargo install solana-vanity

# Copy the worker
COPY worker.py .

# Install common libs just in case worker.py needs them (optional but safe)
RUN pip3 install requests cryptography base58 || true

CMD ["python3", "worker.py"]
