FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
# Note: We rely on .dockerignore to exclude dataset/ and .git
COPY . .

# Expose
EXPOSE 8501

# Run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
