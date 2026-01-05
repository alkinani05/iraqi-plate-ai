#!/bin/bash

# ==========================================
# IRAQI PLATE AI - VPS DEPLOYER
# ==========================================

VPS_IP="45.156.84.161"
VPS_USER="root"
SSH_KEY="/tmp/deploy_key" # Using the secure temp key

echo "🚀 Deploying to ${VPS_IP}..."

# 1. Sync Files (Exclude heavy/useless items)
echo "📂 Syncing Codebase..."
rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'collected_dataset' \
    --exclude '*.mp4' \
    --exclude 'venv' \
    ./ ${VPS_USER}@${VPS_IP}:/root/iraqi-plate-ai/

# 2. Remote Setup execution
echo "🔧 Executing Remote Setup..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${VPS_USER}@${VPS_IP} << 'ENDSSH'
    cd /root/iraqi-plate-ai
    
    # Make setup script executable
    chmod +x scripts/setup_production.sh
    
    # Run the One-Click Setup
    # Note: This handles Docker, Nginx, and SSL automatically
    ./scripts/setup_production.sh
ENDSSH

echo "✅ SUCCESS: App is Live!"
