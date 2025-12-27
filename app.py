
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

# Initialize Session State early
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ---------------------------------------------------------------------
# 🎨 CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;700&display=swap');
    
    /* 🌌 ANIMATED DEEP SPACE BACKGROUND */
    .stApp {
        background: linear-gradient(-45deg, #050510, #1a1a2e, #0f0c29, #1b001b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e0e0e0;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* 🔮 GLASSMORPHISM CONTAINERS */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00ff80 !important;
        text-shadow: 0 0 10px rgba(0,255,128,0.5);
    }
    
    /* 📊 STATS CARDS WITH NEON BORDERS */
    .stats-container {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #00ff80; /* Default Green */
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stats-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 255, 128, 0.2);
        border-color: #00ffff; /* Hover Cyan */
    }
    
    /* 🚀 UPLOAD BOX STYLING */
    .stFileUploader {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        background: rgba(0,0,0,0.2);
        transition: border 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #00ff80;
        background: rgba(0, 255, 128, 0.05);
    }

    /* 🖋️ TYPOGRAPHY */
    h1 {
        font-family: 'Share Tech Mono', monospace;
        color: #fff;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    h3 {
        font-family: 'Outfit', sans-serif;
        color: #00ffff; /* Cyan Headers */
        border-bottom: 2px solid rgba(0, 255, 255, 0.3);
        padding-bottom: 10px;
        margin-top: 20px;
    }
    
    /* ✨ PULSE ANIMATION FOR LOGO */
    @keyframes pulse {
        0% { opacity: 0.8; text-shadow: 0 0 20px rgba(0,255,128,0.6); }
        50% { opacity: 1.0; text-shadow: 0 0 40px rgba(0,255,128,0.9), 0 0 80px rgba(0,255,128,0.4); }
        100% { opacity: 0.8; text-shadow: 0 0 20px rgba(0,255,128,0.6); }
    }
    
    .stat-number {
        font-size: 3.5rem;
        font-weight: bold;
        background: -webkit-linear-gradient(#00ff80, #008040);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Outfit', sans-serif;
    }
    
    /* 🔴 ALERT BOXES */
    .stAlert {
        background-color: rgba(0,0,0,0.6) !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
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
# ---------------------------------------------------------------------
# TAB 1: Photo Upload
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# TAB 1: Photo Upload
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Upload Iraqi License Plate Photo")
    
    # Form for reliable mobile submission
    with st.form("upload_form"):
        uploaded_photo = st.file_uploader("📸 Take Photo / 📂 Upload Image", type=['jpg', 'jpeg', 'png'])
        submitted = st.form_submit_button("🚀 ANALYZE IMAGE", use_container_width=True)
    
    if submitted and uploaded_photo:
        # Create unique ID for this upload
        file_id = f"{uploaded_photo.name}_{uploaded_photo.size}"
        
        # Show preview
        st.image(uploaded_photo, caption="Preview", use_column_width=True)
        
        #  SEAMLESS PROCESSING (No Button Required)
        try:
            from PIL import Image, ImageOps
            uploaded_photo.seek(0)
            image_pil = Image.open(uploaded_photo)
            # 🔄 Fix Mobile Rotation (EXIF)
            image_pil = ImageOps.exif_transpose(image_pil)
            # Convert PIL (RGB) to OpenCV (BGR)
            img = np.array(image_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
             st.error(f"❌ Error loading image: {e}")
             img = None
        
        if img is None:
            st.error("❌ Error: Could not understand image format.")
        else:
            with st.spinner("🤖 Analyzing..."):
                results = reader.predict(img)
                viz = reader.visualize(img, results)
                
                st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Analysis Result")
                
                # Smart Processing: Only save/stats if new file
                is_new_file = file_id not in st.session_state.processed_files
                
                if results:
                    st.success(f"✅ Found {len(results)} plates!")
                    
                    if is_new_file:
                        batch_id = datetime.now().strftime("%H%M%S") + "_" + hashlib.md5(file_id.encode()).hexdigest()[:6]
                        
                        for res in results:
                            conf = res['conf']
                            text = res['text']
                            x1, y1, x2, y2 = res['box']
                            plate_crop = img[y1:y2, x1:x2]
                            
                            st.write(f"**{text}** ({int(conf*100)}%)")
                            if save_entry(img, plate_crop, text, conf, batch_id):
                                stats['plates_captured'] += 1
                        
                        stats['total_uploads'] += 1
                        save_stats(stats)
                        st.session_state.processed_files.add(file_id)
                        
                        st.balloons()
                    else:
                        st.caption("ℹ️ This image has already been processed and saved.")
                        for res in results:
                             st.write(f"**{res['text']}** ({int(res['conf']*100)}%)")

                else:
                    st.info("No plates detected")
                    if is_new_file:
                        stats['total_uploads'] += 1
                        save_stats(stats)
                        st.session_state.processed_files.add(file_id)
    elif submitted and not uploaded_photo:
        st.warning("⚠️ Please select or take a photo first!")

# ---------------------------------------------------------------------
# TAB 2: Video Upload
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# TAB 2: Video Upload
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Process Iraqi License Plate Video")
    uploaded_video = st.file_uploader("Upload Video File (MP4/MOV)", type=['mp4', 'avi', 'mov'], key="video")
    
    if uploaded_video:
        st.info("Video uploaded! Click below to start processing.")
        if st.button("🎬 START PROCESSING VIDEO", use_container_width=True):
            # Save temp video
            target_video_path = "temp_upload.mp4"
            with open(target_video_path, 'wb') as f:
                f.write(uploaded_video.read())
            
            with st.spinner("🎬 Processing video frames..."):
                cap = cv2.VideoCapture(target_video_path)
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
                if os.path.exists(target_video_path):
                    os.remove(target_video_path)
                
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
# ☁️ CLOUD SYNC LOGIC
# ---------------------------------------------------------------------
def sync_to_cloud():
    """Uploads collected_dataset to Hugging Face Hub"""
    from huggingface_hub import HfApi
    # SECURITY FIX: Use Secret from Environment
    TOKEN = os.environ.get("HF_TOKEN")
    if not TOKEN:
        return False, "❌ Error: HF_TOKEN secret not set in Space settings!"
    DATASET_REPO = "husam05/iraqi-plate-dataset"
    
    api = HfApi(token=TOKEN)
    
    try:
        api.upload_folder(
            folder_path=str(DATASET_DIR),
            repo_id=DATASET_REPO,
            repo_type="dataset",
            path_in_repo="data", # Store inside a data/ folder
            commit_message=f"Auto-sync: {datetime.now().isoformat()}"
        )
        return True, "✅ Synced successfully!"
    except Exception as e:
        return False, f"❌ Sync Failed: {e}"

# ---------------------------------------------------------------------
# TAB 3: Admin Panel
# ---------------------------------------------------------------------
with tab3:
    if check_password():
        st.success(f"✅ Logged in as {ADMIN_USER}")
        
        # Dashboard Controls
        st.markdown("### ☁️ Cloud Data Management")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            if st.button("☁️ SYNC TO CLOUD", use_container_width=True):
                with st.spinner("Uploading data to 'husam05/iraqi-plate-dataset'..."):
                    success, msg = sync_to_cloud()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        
        with col_ctrl2:
            # Download Dataset Button
            if st.button("📦 Download ZIP", use_container_width=True):
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
                    label="⬇️ Click to Save ZIP",
                    data=zip_buffer,
                    file_name=f"iraqi_plates_dataset_v4.2_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        with col_ctrl3:
             st.metric("Pending Local Files", len(list(CROPS_DIR.glob("*.jpg"))))

        st.markdown("---")
        
        # Data Management Header
        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
             st.markdown("### 🛠️ Review & Clean Data")
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
                    if st.button("🗑️ DELETE", key=f"del_{img_path.name}", use_container_width=True):
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

