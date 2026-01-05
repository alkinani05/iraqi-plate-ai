#!/bin/bash
VPS_IP="45.156.84.161"
VPS_USER="root"

echo "🔧 Installing Docker-Compose & Fixing on ${VPS_IP}..."

ssh -t ${VPS_USER}@${VPS_IP} << 'ENDSSH'
    # 1. Ensure Docker is running
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found! Installing..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
    fi

    # 2. Try to install docker-compose binary if missing
    if ! command -v docker-compose &> /dev/null; then
        echo "⬇️ Installing standalone docker-compose..."
        curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
    fi

    # 3. Permissions
    echo "🔑 Fixing permissions..."
    mkdir -p /root/iraqi-plate-ai/collected_dataset
    chmod -R 777 /root/iraqi-plate-ai/collected_dataset

    # 4. Run App
    echo "🚀 Launching App..."
    cd /root/iraqi-plate-ai
    
    # Try multiple ways to start
    if command -v docker-compose &> /dev/null; then
        docker-compose down || true
        docker-compose up -d --build
    else
        echo "⚠️ docker-compose failed, trying 'docker compose'..."
        docker compose down || true
        docker compose up -d --build
    fi

    echo "⏳ Waiting for logs..."
    sleep 5
    docker ps
    docker logs --tail 20 iraqi-plate-collector || docker logs --tail 20 iraqi-plate-ai-plate-collector-1
ENDSSH
