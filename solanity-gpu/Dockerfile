FROM ubuntu:22.04

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Ensure Python logs are output immediately
ENV PYTHONUNBUFFERED=1

# Install Python, Pip, and dependencies
RUN apt-get update && apt-get install -y \
    curl python3 python3-pip bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Official Solana CLI (Stable Release)
# Using direct download to avoid pipe errors
RUN curl -L https://github.com/solana-labs/solana/releases/download/v1.18.4/solana-release-x86_64-unknown-linux-gnu.tar.bz2 | tar jx -C /tmp && \
    mv /tmp/solana-release/bin/solana-keygen /usr/local/bin/solana-keygen && \
    rm -rf /tmp/solana-release && \
    solana-keygen --version

# Copy the worker script
COPY worker.py .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python3", "worker.py"]
