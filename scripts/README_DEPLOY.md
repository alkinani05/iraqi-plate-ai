# Deploying to Your VPS

Since you want to move away from Hugging Face Spaces and run this on your own VPS with real-time camera support, follow these steps.

**Important**: Modern browsers (Chrome, Safari, Mobile) **BLOCK** camera access on `http://` sites unless they are `localhost`.
To use the live camera on a VPS, you **MUST** serve the site over **HTTPS**.

## Option A: Easy Setup (No Domain, HTTP only)
*Note: "Live Camera" tab might NOT work on phones/Chrome without HTTPS. Upload tabs will work fine.*

1. **Install Docker & Docker Compose** on your VPS:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   ```

2. **Upload your code** to the VPS (e.g., using `scp` or `git clone`).

3. **Run the container**:
   ```bash
   # Set your Hugging Face Token (if you want cloud sync)
   export HF_TOKEN="your_token_here"
   
   # Start
   sudo docker-compose up -d --build
   ```

4. Access at `http://YOUR_VPS_IP:8501`.

---

## Option B: Professional Setup (HTTPS + Domain) - **Recommended**

To make the camera work everywhere, you need a domain (e.g., `plates.yourdomain.com`) and SSL.

### 1. Prerequisites
- A domain name pointing to your VPS IP.
- Nginx installed on VPS (`sudo apt install nginx`).
- Certbot installed (`sudo apt install certbot python3-certbot-nginx`).

### 2. Configure Nginx
Create a config file: `/etc/nginx/sites-available/plate-app`

```nginx
server {
    server_name plates.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable it:
```bash
sudo ln -s /etc/nginx/sites-available/plate-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 3. Get SSL (HTTPS)
Run Certbot:
```bash
sudo certbot --nginx -d plates.yourdomain.com
```

### 4. Run the App
```bash
sudo docker-compose up -d --build
```

Now access `https://plates.yourdomain.com`. The camera will work perfectly!
