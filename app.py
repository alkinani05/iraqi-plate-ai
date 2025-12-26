
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
# 📁 DATASET MANAGEMENT
# ---------------------------------------------------------------------
DATASET_DIR = Path("collected_dataset")
LOW_CONF_DIR = DATASET_DIR / "low_confidence"
STATS_FILE = DATASET_DIR / "stats.json"

def init_dataset():
    """Initialize dataset directories and stats file"""
    DATASET_DIR.mkdir(exist_ok=True)
    LOW_CONF_DIR.mkdir(exist_ok=True)
    
    if not STATS_FILE.exists():
        stats = {
            "total_uploads": 0,
            "photos_collected": 0,
            "videos_processed": 0,
            "last_updated": datetime.now().isoformat()
        }
        save_stats(stats)
    return load_stats()

def load_stats():
    """Load statistics"""
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {"total_uploads": 0, "photos_collected": 0, "videos_processed": 0}

def save_stats(stats):
    """Save statistics"""
    stats["last_updated"] = datetime.now().isoformat()
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def save_to_dataset(img, text, confidence):
    """Save image to dataset if confidence < 90%"""
    if confidence >= 0.90:
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"plate_{timestamp}_conf{int(confidence*100)}.jpg"
    filepath = LOW_CONF_DIR / filename
    
    cv2.imwrite(str(filepath), img)
    
    # Save metadata
    metadata_file = LOW_CONF_DIR / "metadata.csv"
    with open(metadata_file, 'a') as f:
        f.write(f"{filename},{text},{confidence:.3f},{datetime.now().isoformat()}\n")
    
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
st.markdown("<h1>🚗 IRAQI PLATE COLLECTOR</h1>", unsafe_allow_html=True)

# Public Stats Display
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-number'>{stats['total_uploads']}</div>
        <div class='stat-label'>Total Uploads</div>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-number'>{stats['photos_collected']}</div>
        <div class='stat-label'>Dataset Images</div>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-number'>{stats['videos_processed']}</div>
        <div class='stat-label'>Videos Processed</div>
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
        
        with st.spinner("🤖 Analyzing..."):
            results = reader.predict(img)
            viz = reader.visualize(img, results)
            
            col_img, col_res = st.columns([2, 1])
            
            with col_img:
                st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col_res:
                if results:
                    for res in results:
                        conf = res['conf']
                        text = res['text']
                        
                        if conf >= 0.90:
                            st.success(f"✅ **{text}**\n\nConfidence: {int(conf*100)}%")
                        else:
                            st.warning(f"⚠️ **{text}**\n\nConfidence: {int(conf*100)}%\n\n📥 Saved to dataset")
                        
                        # Only auto-save if new file
                        if file_id not in st.session_state.processed_files:
                            x1, y1, x2, y2 = res['box']
                            plate_crop = img[y1:y2, x1:x2]
                            if save_to_dataset(plate_crop, text, conf):
                                stats['photos_collected'] += 1
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
                        
                        for res in results:
                            conf = res['conf']
                            text = res['text']
                            x1, y1, x2, y2 = res['box']
                            plate_crop = frame[y1:y2, x1:x2]
                            
                            if save_to_dataset(plate_crop, text, conf):
                                collected_count += 1
                        
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                
                st.success(f"✅ Video processed!\n\n📥 Collected {collected_count} uncertain plates for dataset")
                
                # Update stats
                stats['total_uploads'] += 1
                stats['videos_processed'] += 1
                stats['photos_collected'] += collected_count
                save_stats(stats)
                st.session_state.processed_files.add(file_id)
                st.rerun()
        else:
            st.info("✅ Video already processed. Check stats above.")

# ---------------------------------------------------------------------
# TAB 3: Admin Panel
# ---------------------------------------------------------------------
with tab3:
    if check_password():
        st.success(f"✅ Logged in as {ADMIN_USER}")
        
        # Download Dataset Button
        if st.button("📦 Download Full Dataset"):
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file in LOW_CONF_DIR.glob("*"):
                    zip_file.write(file, file.name)
            
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer,
                file_name=f"iraqi_plates_dataset_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip"
            )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Preview")
        
        # Display collected images
        images = list(LOW_CONF_DIR.glob("*.jpg"))
        if images:
            cols = st.columns(4)
            for idx, img_path in enumerate(images[:20]):  # Show first 20
                with cols[idx % 4]:
                    img = cv2.imread(str(img_path))
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=img_path.name, use_column_width=True)
            
            if len(images) > 20:
                st.info(f"Showing 20 of {len(images)} images. Download full dataset to see all.")
        else:
            st.info("No data collected yet. Upload photos/videos with confidence < 90%.")
        
        # Reset Dataset (Danger Zone)
        st.markdown("---")
        st.markdown("### ⚠️ Danger Zone")
        if st.button("🗑️ Reset All Data", type="secondary"):
            if st.checkbox("I confirm deletion"):
                import shutil
                shutil.rmtree(DATASET_DIR)
                init_dataset()
                st.success("Dataset cleared!")
                st.rerun()

