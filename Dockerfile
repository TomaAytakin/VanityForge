FROM rust:latest as builder
WORKDIR /app
COPY cpu-grinder/ .
RUN apt-get update && apt-get install -y pkg-config libssl-dev
RUN cargo build --release

FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir firebase_admin google-cloud-firestore cryptography base58
COPY --from=builder /app/target/release/cpu-grinder ./cpu-grinder-bin
COPY worker_cpu.py .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "worker_cpu.py"]
