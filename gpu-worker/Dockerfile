FROM rust:1.75-slim-bookworm

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    pkg-config \
    libssl-dev \
    udev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Cargo files and source code
COPY Cargo.toml .
COPY src/ src/

# Build the release binary
RUN cargo build --release

# Copy Python requirements and install them
COPY requirements.txt .
RUN pip3 install -r requirements.txt --break-system-packages

# Copy the worker script
COPY worker.py .

# Run the worker script
CMD ["python3", "worker.py"]
