# Stage 1: Build Rust Miner (Updated to latest Rust)
FROM rust:latest as builder
WORKDIR /app
COPY cpu-grinder/ .
# Install dependencies needed for compilation
RUN apt-get update && apt-get install -y pkg-config libssl-dev
RUN cargo build --release

# Stage 2: Runtime (Python + Rust Binary)
FROM python:3.9-slim
WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir firebase_admin google-cloud-firestore

# Copy compiled binary from Stage 1
COPY --from=builder /app/target/release/cpu-grinder ./cpu-grinder-bin

# Copy Python worker script
COPY worker_cpu.py .

CMD ["python", "worker_cpu.py"]
