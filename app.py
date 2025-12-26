
import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
import zipfile
import io

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_plate_reader import PlateReader

# ---------------------------------------------------------------------
# 🔐 AUTHENTICATION
# ---------------------------------------------------------------------
ADMIN_USER = "husam"
ADMIN_PASS_HASH = hashlib.sha256("987987987".encode()).hexdigest()

def check_password():
    """Returns True if password is correct"""
    def password_entered():
        if (st.session_state["username"] == ADMIN_USER and 
            hashlib.sha256(st.session_state["password"].encode()).hexdigest() == ADMIN_PASS_HASH):
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("🔒 Incorrect credentials")
        return False
    else:
        return True

# ---------------------------------------------------------------------
# 📁 DATASET MANAGEMENT v4.2
# ---------------------------------------------------------------------
DATASET_DIR = Path("collected_dataset")
FULL_IMG_DIR = DATASET_DIR / "full_images"
CROPS_DIR = DATASET_DIR / "crops"
METADATA_FILE = DATASET_DIR / "metadata.csv"
STATS_FILE = DATASET_DIR / "stats.json"

def init_dataset():
    """Initialize dataset directories and stats file"""
    # Create new structure
    DATASET_DIR.mkdir(exist_ok=True)
    FULL_IMG_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)
    
    # Initialize metadata header if new
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, 'w') as f:
            f.write("timestamp,full_image_id,crop_image_id,predicted_text,confidence,review_status\n")
    
    if not STATS_FILE.exists():
        stats = {
            "total_uploads": 0,
            "plates_captured": 0,
            "last_updated": datetime.now().isoformat()
        }
        save_stats(stats)
    return load_stats()

def load_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {"total_uploads": 0, "plates_captured": 0}

def save_stats(stats):
    stats["last_updated"] = datetime.now().isoformat()
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def save_entry(full_img, plate_crop, text, confidence, base_filename):
    """
    Save Full Image + Crop + Metadata
    base_filename: unique ID for the capture (e.g. timestamp_uuid)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save Full Image (if not already saved for this batch)
    # We use the base_filename to check existence to avoid duplicates if multiple plates in one car
    full_img_name = f"full_{base_filename}.jpg"
    full_img_path = FULL_IMG_DIR / full_img_name
    if not full_img_path.exists():
        cv2.imwrite(str(full_img_path), full_img)
        
    # 2. Save Crop
    crop_name = f"crop_{base_filename}_{text}.jpg"
    # Sanitize text for filename
    safe_text = "".join([c for c in text if c.isalnum()])
    crop_name = f"crop_{base_filename}_{safe_text}.jpg"
    
    cv2.imwrite(str(CROPS_DIR / crop_name), plate_crop)
    
    # 3. Log to CSV
    # format: timestamp, full_img_name, crop_name, text, confidence, [PENDING]
    with open(METADATA_FILE, 'a') as f:
        f.write(f"{timestamp},{full_img_name},{crop_name},{text},{confidence:.4f},PENDING\n")
    
    return True

# ---------------------------------------------------------------------
# ⚙️ CONFIG & PAGE SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Iraqi Plate Collector", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------
# 🎨 CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    h1 {
        font-family: 'Share Tech Mono', monospace;
        color: #00ff80;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0,255,128,0.5);
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; text-shadow: 0 0 20px rgba(0,255,128,0.6); }
        50% { opacity: 1.0; text-shadow: 0 0 40px rgba(0,255,128,0.9); }
        100% { opacity: 0.8; text-shadow: 0 0 20px rgba(0,255,128,0.6); }
    }
    
    .stats-container {
        background: rgba(0, 255, 128, 0.05);
        border: 2px solid #00ff80;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff80;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 🧠 AI LOGIC
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return PlateReader()

try:
    reader = load_model()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# Initialize dataset
stats = init_dataset()

# ---------------------------------------------------------------------
# 📱 MAIN APP
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 📱 MAIN APP
# ---------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="margin-bottom: 0;">DATA COLLECTOR v4.2</h1>
    <div style="
        font-family: 'Outfit', sans-serif;
        color: #666;
        font-size: 1rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        margin-top: 10px;
    ">
        POWERED BY <span style="
            color: #00ff80; 
            font-weight: 700; 
            text-shadow: 0 0 20px rgba(0,255,128,0.6);
            animation: pulse 2s infinite;
        ">ALGONEST AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Public Stats Display
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-number'>{stats['total_uploads']}</div>
        <div class='stat-label'>Contributions</div>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-number'>{stats['plates_captured']}</div>
        <div class='stat-label'>Plates In Database</div>
    </div>
    """, unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📸 UPLOAD PHOTO", "🎬 UPLOAD VIDEO", "🔐 ADMIN"])

# ---------------------------------------------------------------------
# TAB 1: Photo Upload
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Upload Iraqi License Plate Photo")
    uploaded_photo = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="photo")
    
    if uploaded_photo:
        # Create unique ID for this upload
        file_id = f"{uploaded_photo.name}_{uploaded_photo.size}"
        
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
            
        file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("🤖 Analyzing & Archiving..."):
            results = reader.predict(img)
            viz = reader.visualize(img, results)
            
            col_img, col_res = st.columns([2, 1])
            
            with col_img:
                st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col_res:
                if results:
                    st.success(f"✅ Found {len(results)} plates!")
                    
                    # Generate a unique batch ID for this image
                    # Use timestamp + hash of filename to be unique but deterministic
                    batch_id = datetime.now().strftime("%H%M%S") + "_" + hashlib.md5(file_id.encode()).hexdigest()[:6]
                    
                    for res in results:
                        conf = res['conf']
                        text = res['text']
                        x1, y1, x2, y2 = res['box']
                        plate_crop = img[y1:y2, x1:x2]
                        
                        st.write(f"**{text}** ({int(conf*100)}%)")
                        
                        # Only auto-save if new file
                        if file_id not in st.session_state.processed_files:
                            if save_entry(img, plate_crop, text, conf, batch_id):
                                stats['plates_captured'] += 1
                else:
                    st.info("No plates detected")
                
                # Only update stats once per file
                if file_id not in st.session_state.processed_files:
                    stats['total_uploads'] += 1
                    save_stats(stats)
                    st.session_state.processed_files.add(file_id)
                    st.rerun()

# ---------------------------------------------------------------------
# TAB 2: Video Upload
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Upload Iraqi License Plate Video")
    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'], key="video")
    
    if uploaded_video:
        file_id = f"{uploaded_video.name}_{uploaded_video.size}"
        
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
            
        # Only process if new
        if file_id not in st.session_state.processed_files:
            # Save temp video
            temp_video = "temp_upload.mp4"
            with open(temp_video, 'wb') as f:
                f.write(uploaded_video.read())
            
            with st.spinner("🎬 Processing video..."):
                cap = cv2.VideoCapture(temp_video)
                frame_count = 0
                collected_count = 0
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0: total_frames = 1
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process every 10th frame
                    if frame_count % 10 == 0:
                        results = reader.predict(frame)
                        
                        if results:
                            # Unique batch ID for this frame highlight
                            batch_id = f"vid_{datetime.now().strftime('%H%M%S')}_f{frame_count}"
                            
                            for res in results:
                                conf = res['conf']
                                text = res['text']
                                x1, y1, x2, y2 = res['box']
                                plate_crop = frame[y1:y2, x1:x2]
                                
                                if save_entry(frame, plate_crop, text, conf, batch_id):
                                    collected_count += 1
                        
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                
                st.success(f"✅ Video Processed! Archived {collected_count} plates.")
                
                # Update stats
                stats['total_uploads'] += 1
                stats['plates_captured'] += collected_count
                save_stats(stats)
                st.session_state.processed_files.add(file_id)
                st.rerun()
        else:
            st.info("✅ Video already processed. Check stats above.")

# ---------------------------------------------------------------------
# TAB 3: Admin Panel
# ---------------------------------------------------------------------
def delete_entry(crop_filename):
    """Delete a specific entry from filesystem and metadata"""
    # 1. Delete Crop File
    crop_path = CROPS_DIR / crop_filename
    if crop_path.exists():
        os.remove(crop_path)
    
    # 2. Update Metadata CSV
    if METADATA_FILE.exists():
        lines = []
        with open(METADATA_FILE, 'r') as f:
            lines = f.readlines()
        
        with open(METADATA_FILE, 'w') as f:
            for line in lines:
                # Check if this line contains the filename
                if crop_filename not in line:
                    f.write(line)
    
    # 3. Update Stats
    stats = load_stats()
    if stats['plates_captured'] > 0:
        stats['plates_captured'] -= 1
        save_stats(stats)
    
    return True

# ---------------------------------------------------------------------
# TAB 3: Admin Panel
# ---------------------------------------------------------------------
with tab3:
    if check_password():
        st.success(f"✅ Logged in as {ADMIN_USER}")
        
        col_main1, col_main2 = st.columns([1, 1])
        with col_main1:
            # Download Dataset Button
            if st.button("📦 Download Full Dataset (Images + Metadata)"):
                # Create ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Walk through the directory and add all files
                    for root, dirs, files in os.walk(DATASET_DIR):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, DATASET_DIR)
                            zip_file.write(file_path, arcname)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer,
                    file_name=f"iraqi_plates_dataset_v4.2_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip"
                )

        st.markdown("---")
        
        # Data Management Header
        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
             st.markdown("### 🛠️ Manage Data")
        with col_head2:
             show_all = st.checkbox("Show All Entries")

        # Display collected crops with DELETE button
        images = list(CROPS_DIR.glob("*.jpg"))
        # Sort by newest first (modification time)
        images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if images:
            limit = len(images) if show_all else 24
            
            # Create grid
            cols = st.columns(6)
            for idx, img_path in enumerate(images[:limit]):
                with cols[idx % 6]:
                    img = cv2.imread(str(img_path))
                    # Display Image
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    # Display Filename (truncated)
                    st.caption(f"{img_path.name[:10]}...")
                    # Delete Button
                    if st.button("🗑️ DELETE", key=f"del_{img_path.name}"):
                        delete_entry(img_path.name)
                        st.rerun()
            
            if not show_all and len(images) > 24:
                st.info(f"Showing 24 of {len(images)} most recent crops. Check 'Show All Entries' to see everything.")
        else:
            st.info("No data collected yet.")
        
        # Reset Dataset (Danger Zone)
        st.markdown("---")
        st.markdown("### ⚠️ Danger Zone")
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            if st.button("🗑️ DELETE EVERYTHING", type="secondary"):
                if st.checkbox("I confirm complete deletion"):
                    import shutil
                    if DATASET_DIR.exists():
                        shutil.rmtree(DATASET_DIR)
                    init_dataset()
                    st.success("Dataset CLEARED! ✅")
                    st.rerun()

