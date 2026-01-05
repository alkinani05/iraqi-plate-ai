#!/bin/bash

# ==========================================
# IRAQI PLATE AI - DEPLOY (INTERACTIVE)
# ==========================================

VPS_IP="45.156.84.161"
VPS_USER="root"

echo "🚀 Deploying to $VPS_IP..."
echo "⚠️  You will be asked for the VPS Password TWICE."

# 1. Sync
echo "📂 [1/2] Syncing Code (Enter Password)..."
rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude 'collected_dataset' \
    --exclude '*.mp4' \
    --exclude 'venv' \
    ./ ${VPS_USER}@${VPS_IP}:/root/iraqi-plate-ai/

# 2. Setup
echo "🔧 [2/2] Running Setup (Enter Password)..."
ssh -o StrictHostKeyChecking=no ${VPS_USER}@${VPS_IP} "cd /root/iraqi-plate-ai && chmod +x scripts/setup_production.sh && ./scripts/setup_production.sh"

echo "✅ SUCCESS! Check http://$VPS_IP:8501"
