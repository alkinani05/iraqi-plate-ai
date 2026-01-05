#!/bin/bash

# Define VPS Details
VPS_IP="45.156.84.161"
VPS_USER="root"
REMOTE_DIR="/root/iraqi-plate-ai"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🚀 Starting Deployment to ${VPS_IP}...${NC}"

# 1. Prepare Docker image locally (save transfer time if internet is slow there, but building there is usually better)
# We will transfer source and build THERE to ensure architecture compatibility (linux/amd64)

echo -e "${CYAN}📦 Uploading files (Enter Password if asked)...${NC}"
# Exclude venv, .git, and bulky dataset for initial sync if not needed
# Using rsync if available, else scp
if command -v rsync &> /dev/null; then
    rsync -avz --exclude 'venv' --exclude '.git' --exclude 'collected_dataset' --exclude '__pycache__' . ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}
else
    ssh -i jet.pri -o StrictHostKeyChecking=no ${VPS_USER}@${VPS_IP} "mkdir -p ${REMOTE_DIR}"
    scp -i jet.pri -o StrictHostKeyChecking=no -r app.py requirements.txt Dockerfile docker-compose.yml smart_plate_reader.py ip_scanner.py detector.pt professional_real_only.pth fonts ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}
fi

echo -e "${GREEN}✅ Files Uploaded.${NC}"

# 2. Remote Setup & Launch
echo -e "${CYAN}🔧 Configuring VPS & Launching Docker...${NC}"

ssh -i jet.pri -o StrictHostKeyChecking=no -t ${VPS_USER}@${VPS_IP} << 'ENDSSH'
    # Update and Install Docker if missing
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        apt-get update && apt-get install -y docker.io docker-compose
    fi

    cd /root/iraqi-plate-ai

    # Create dataset dir to persist data
    mkdir -p collected_dataset

    # Build and Run
    echo "🐳 Building Docker Container..."
    docker-compose up -d --build

    echo "✅ DEPLOYMENT COMPLETE!"
    echo "docker ps"
    docker ps
ENDSSH

echo -e "${GREEN}✨ App should be live at http://${VPS_IP}:8501 ${NC}"
