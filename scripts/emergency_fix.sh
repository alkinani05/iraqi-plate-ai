#!/bin/bash
VPS_IP="45.156.84.161"
VPS_USER="root"

echo "🔧 EMERGENCY DOCKER FIX on ${VPS_IP}..."

ssh -t ${VPS_USER}@${VPS_IP} << 'ENDSSH'
    echo "1. Removing old Docker versions..."
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do apt-get remove $pkg; done

    echo "2. Installing Latest Docker Official..."
    # Add Docker's official GPG key:
    apt-get update
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up the repository:
    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "3. RUNNING APP VIA 'docker compose' (plugin)..."
    cd /root/iraqi-plate-ai
    
    # Prune old junk
    docker container prune -f
    
    # Launch
    docker compose up -d --build

    echo "✅ DONE."
    docker ps
ENDSSH
