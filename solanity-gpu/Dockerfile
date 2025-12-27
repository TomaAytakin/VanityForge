FROM ubuntu:22.04

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Pip, and dependencies
# We need 'base58' to format the private key for the user
RUN apt-get update && apt-get install -y \
    curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install the 'base58' library for Python (crucial for formatting)
RUN pip3 install base58

WORKDIR /app

# Install Official Solana CLI (Stable Release)
RUN sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"

# Add Solana binaries to PATH
ENV PATH="/root/.local/share/solana/install/active_release/bin:${PATH}"

# Copy the worker script
COPY worker.py .

CMD ["python3", "worker.py"]