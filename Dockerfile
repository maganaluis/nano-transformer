# Stage 1: Builder
FROM python:3.13-bullseye AS builder

# Install Python 3, venv, pip, and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and create a virtual environment
COPY . .
RUN python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.13-bullseye

# Install Python in the runtime stage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy the application code from the builder stage
COPY --from=builder /app /app

# Ensure that the virtual environment is used
ENV PATH="/venv/bin:$PATH"

# Set the command to run your application
CMD ["python3", "main.py"]
