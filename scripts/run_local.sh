#!/bin/bash

# Iraqi Plate AI - Local Runner
# Powered by ALGONEST AI

echo "🚀 Starting Iraqi Plate Collector (Local Mode)..."

# 1. Check for Models
if [ ! -f "detector.pt" ] || [ $(wc -c <"detector.pt") -lt 1000 ]; then
    echo "⬇️ Downloading AI Models (this may take a minute)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
try:
    path = hf_hub_download(repo_id='husam05/iraqi-plate-ai', filename='detector.pt', repo_type='space')
    shutil.copy(path, 'detector.pt')
    print('✅ detector.pt downloaded')
except Exception as e: print(e)
"
fi

if [ ! -f "professional_real_only.pth" ] || [ $(wc -c <"professional_real_only.pth") -lt 1000 ]; then
     python3 -c "
from huggingface_hub import hf_hub_download
import shutil
try:
    path = hf_hub_download(repo_id='husam05/iraqi-plate-ai', filename='professional_real_only.pth', repo_type='space')
    shutil.copy(path, 'professional_real_only.pth')
    print('✅ professional_real_only.pth downloaded')
except Exception as e: print(e)
"
fi

# 2. Install Deps if needed
if ! pip show streamlit > /dev/null; then
    echo "📦 Installing Dependencies..."
    pip install -r requirements.txt
fi

# 3. Run App
echo "✅ Launching App..."
echo "👉 Open http://localhost:8501 in your browser"
python3 -m streamlit run app.py --server.port 8501
