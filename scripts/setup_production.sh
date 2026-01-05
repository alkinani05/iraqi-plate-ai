#!/bin/bash

# ==========================================
# IRAQI PLATE AI - PRODUCTION SETUP SCRIPT
# ==========================================
# Usage: ./setup_production.sh [--update]

set -e

APP_DIR="/root/iraqi-plate-ai"
DOMAIN="plates.yourdomain.com" # Change this!
EMAIL="admin@yourdomain.com"   # Change this!

echo "🚀 Starting Production Setup..."

# 1. Update System & Install Dependencies
echo "📦 Installing System Dependencies..."
apt-get update
# Ensure Docker is latest
if ! command -v docker &> /dev/null; then
    echo "   - Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    echo "   - Docker already installed."
fi

# Install Nginx & Certbot
apt-get install -y nginx certbot python3-certbot-nginx

# 2. Firewall Setup (UFW)
echo "🛡️ Configuring Firewall..."
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw allow 8501/tcp # Optional: if you want direct access without Nginx
# ufw enable # CAREFUL: Don't lock yourself out!

# 3. Application Setup
echo "📂 Setting up Application Directory..."
if [ ! -d "$APP_DIR" ]; then
    echo "   - Directory not found. Please clone repo first to $APP_DIR"
    exit 1
fi

cd $APP_DIR

# Create Directories
mkdir -p collected_dataset/full_images
mkdir -p collected_dataset/crops
mkdir -p collected_dataset/db
chmod -R 777 collected_dataset

# 4. Nginx Configuration (HTTPS)
echo "🌐 Configuring Nginx Reverse Proxy..."
NGINX_CONF="/etc/nginx/sites-available/plate-app"

cat > $NGINX_CONF <<EOF
server {
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOF

# Enable Site
ln -sf $NGINX_CONF /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# 5. SSL Certificate (Interactive if not exists)
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    echo "🔒 Requesting SSL Certificate (Certbot)..."
    echo "   (Skipping in script mode. Run manually: certbot --nginx -d $DOMAIN)"
fi

# 6. Docker Deployment
echo "🐳 Building & Deploying Containers..."
docker compose down || true
docker compose up -d --build

echo "✅ Deployment Complete!"
echo "   - App URL: https://$DOMAIN"
echo "   - Local:   http://localhost:8501"
