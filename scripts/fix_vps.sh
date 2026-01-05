#!/bin/bash
VPS_IP="45.156.84.161"
VPS_USER="root"

echo "🔧 Running Diagnostics & Fix on ${VPS_IP}..."

# Use identity file if available to key skip password
SSH_OPT=""
if [ -f "jet.pri" ]; then
    chmod 600 jet.pri
    SSH_OPT="-i jet.pri"
fi

ssh $SSH_OPT -o StrictHostKeyChecking=no -t ${VPS_USER}@${VPS_IP} << 'ENDSSH'
    echo "1️⃣  Fixing Docker Version Mismatch..."
    # The VPS has a very new Docker Daemon (API 1.44+) but an old Client (1.43)
    # We will manually install a modern standalone Docker Compose binary
    
    echo "   - Downloading Docker Compose v2.27.0..."
    curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    echo "   - Verifying version:"
    docker-compose version

    echo "2️⃣  Fixing Permissions for Database..."
    mkdir -p /root/iraqi-plate-ai/collected_dataset
    chmod -R 777 /root/iraqi-plate-ai/collected_dataset

    echo "3️⃣  Ensuring System Firewall is Open..."
    ufw allow 8501/tcp > /dev/null 2>&1
    iptables -C INPUT -p tcp --dport 8501 -j ACCEPT > /dev/null 2>&1 || iptables -I INPUT -p tcp --dport 8501 -j ACCEPT

    echo "4️⃣  Restarting App Container..."
    cd /root/iraqi-plate-ai
    
    # Clean restart
    docker-compose down || true
    docker-compose up -d --build

    echo "5️⃣  Waiting for startup..."
    sleep 5
    
    echo "📊 CONTAINER LOGS:"
    docker logs --tail 20 iraqi-plate-collector
    
    echo "✅ DONE. Check http://${VPS_IP}:8501"
ENDSSH
