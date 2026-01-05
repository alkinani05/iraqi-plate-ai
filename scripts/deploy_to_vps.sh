#!/bin/bash

# ==========================================
# IRAQI PLATE AI - VPS DEPLOYER (SSH)
# ==========================================
# Usage: ./deploy_to_vps.sh [VPS_IP]

set -e

VPS_IP=${1:-"45.156.84.161"} # Default to user's known VPS
VPS_USER="root"
SSH_KEY_PATH="id_rsa"

# 1. Create Identity File
echo "🔑 Configuring Deploy Key..."
cat > $SSH_KEY_PATH <<EOF
-----BEGIN OPENSSH PRIVATE KEY-----
(You must paste the PRIVATE key here, not the PUBLIC key you sent)
-----END OPENSSH PRIVATE KEY-----
EOF
chmod 600 $SSH_KEY_PATH

echo "⚠️  Wait! You provided a PUBLIC key (starts with 'AAAAB3...')."
echo "   I need the PRIVATE key to connect to the VPS."
echo "   Please check if you have the 'jet.pri' file or the private key content."
echo "   The string you sent is the lock, but I need the key!"

# Cleanup public key usage attempt since it won't work
rm $SSH_KEY_PATH

exit 1
