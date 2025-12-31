# 1. Builder: Use Bookworm (Debian 12) to match Runner
FROM rust:1.75-bookworm as builder
WORKDIR /app
COPY cpu-grinder/ .
RUN cargo build --release

# 2. Runner: Use Bookworm (Debian 12)
FROM python:3.9-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install System Libs for Rust Binary & Python Crypto
RUN apt-get update && \
    apt-get install -y ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*

# Install Python Deps
COPY solanity-gpu/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Artifacts
COPY --from=builder /app/target/release/cpu-grinder ./cpu-grinder-bin
COPY solanity-gpu/worker.py .

# Setup Permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/cpu-grinder-bin

USER appuser

ENV WORKER_TYPE=CPU
ENV BINARY_PATH=/app/cpu-grinder-bin

CMD ["python", "worker.py"]
