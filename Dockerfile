FROM rust:latest as builder
WORKDIR /app
COPY cpu-grinder/ .
# Add curve25519-dalek if not in Cargo.toml (it is now)
RUN cargo build --release

FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y libssl3 ca-certificates

RUN pip install --no-cache-dir firebase_admin google-cloud-firestore cryptography base58 google-cloud-logging requests

# Copy the compiled Rust binary
COPY --from=builder /app/target/release/cpu-grinder ./cpu-grinder-bin

# Copy the UNIFIED worker script from solanity-gpu
COPY solanity-gpu/worker.py .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
RUN chmod +x ./cpu-grinder-bin
USER appuser

# Set environment variable to tell worker to use CPU binary
ENV WORKER_TYPE=CPU
ENV BINARY_PATH=./cpu-grinder-bin

CMD ["python", "worker.py"]
