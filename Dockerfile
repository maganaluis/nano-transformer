FROM python:3.13-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Setup SSHD for MPI over SSH
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Disable SSH host key checking for system-wide SSH clients
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Create SSH client config for root to avoid updating known_hosts
RUN mkdir -p /root/.ssh && \
    echo -e "Host *\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile /dev/null" > /root/.ssh/config

# Expose SSH port for MPI communication
EXPOSE 22

# Setup working directory
WORKDIR /app

# Copy application code
COPY . .

# Set up virtual environment and install dependencies
RUN python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir .

# Use virtual environment in PATH
ENV PATH="/venv/bin:$PATH"

# Default command: start SSH daemon
CMD ["/usr/sbin/sshd", "-D"]
