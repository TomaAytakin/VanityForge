#!/bin/bash
set -e

# Defensive logging
echo "Starting container entrypoint..."
echo "User: $(whoami) (UID: $(id -u))"
echo "Workdir: $(pwd)"

# Ensure output is unbuffered
export PYTHONUNBUFFERED=1

# Check for the binary
if [ -f "./solanity" ]; then
    echo "Binary found: ./solanity"
else
    echo "WARNING: ./solanity binary not found!"
fi

# Check for the worker script
if [ -f "worker.py" ]; then
    echo "Worker script found: worker.py"
else
    echo "ERROR: worker.py not found!"
    exit 1
fi

echo "Exec command: $@"
exec "$@"
