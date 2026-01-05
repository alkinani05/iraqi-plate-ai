
import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
import av
import uuid
import tempfile
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MODULE IMPORTS
from src.core.plate_reader import PlateReader
from src.core.database import Database
from src.utils import network_scanner
from src.ui import auth

# ---------------------------------------------------------------------
# ⚙️ CONFIG & PAGE SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Iraqi Plate Collector Pro", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Load External CSS
def load_css():
    css_path = Path("styles/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------------------
# 🧠 AI & DATA LOGIC
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return PlateReader()

def get_db():
    return Database()

try:
    reader = load_model()
    db = get_db()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 🛠 SIDEBAR TOOLS
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠 Tools / الأدوات")
    st.markdown("---")
    
    st.markdown("#### 📷 Image Enhancement")
    brightness_val = st.slider("Brightness / السطوع", 0.5, 2.0, 1.0, 0.1)
    contrast_val = st.slider("Contrast / التباين", 0.5, 2.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Settings")
    conf_thres = st.slider("Min Confidence", 0.1, 0.9, 0.35, 0.05)
    
    st.markdown("---")
    st.success("GPU/CPU: Active")
    st.info(f"Model: {reader.device}")

def apply_enhancements(img_arr):
    pil_img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness_val)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_val)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------
# 📱 MAIN APP
# ---------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0; text-shadow: 0 0 30px rgba(0, 255, 128, 0.3);">DATA COLLECTOR PRO</h1>
    <div style="font-family: 'Outfit', sans-serif; color: #888; font-size: 1.1rem; letter-spacing: 6px; text-transform: uppercase;">
        POWERED BY <span style="color: #00ff80; font-weight: 800; animation: pulse 2.5s infinite;">ALGONEST AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# TOP KPIs
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.markdown(f"""<div class='stats-container'><div class='stat-number'>{db.get_stat('total_uploads')}</div><div class='stat-label'>Total Uploads / الرفع</div></div>""", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""<div class='stats-container'><div class='stat-number'>{db.get_stat('plates_captured')}</div><div class='stat-label'>Plates Captured / اللوحات</div></div>""", unsafe_allow_html=True)
with kpi3:
    pending_count = db.get_pending_count()
    color = "#ff3333" if pending_count > 0 else "#00ff80"
    st.markdown(f"""<div class='stats-container' style='border-left-color:{color}'><div class='stat-number' style='color:{color}'>{pending_count}</div><div class='stat-label'>Pending Review / قيد المراجعة</div></div>""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 PHOTO", "🎬 VIDEO", "🔴 LIVE CAMERA", "📡 IP CAMS", "🔐 ADMIN"])

# ---------------------------------------------------------------------
# TAB 1: Photo
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### 📸 Photo Analysis")
    uploaded_photo = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_photo:
        if st.button("🚀 ANALYZE", type="primary", use_container_width=True):
            file_id = f"{uploaded_photo.name}_{uploaded_photo.size}"
            
            image_pil = Image.open(uploaded_photo)
            image_pil = ImageOps.exif_transpose(image_pil)
            img = np.array(image_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img = apply_enhancements(img)
            
            with st.spinner("🤖 Processing..."):
                results = reader.predict(img, conf_thres=conf_thres)
                viz = reader.visualize(img, results)
                st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                if results and file_id not in st.session_state.processed_files:
                    batch_id = str(uuid.uuid4())[:8]
                    for res in results:
                         x1, y1, x2, y2 = res['box']
                         crop_img = img[y1:y2, x1:x2]
                         
                         # Save to DB
                         full_name = f"full_{batch_id}.jpg" 
                         crop_name = f"crop_{batch_id}_{int(time.time())}.jpg"
                         
                         db.add_entry(full_name, crop_name, res['text'], res['conf'], batch_id)
                         
                         # Save Disk (Database logic only stores filename, we need actual save)
                         # Wait, DB wrapper expects paths but doesn't write files? 
                         # Let's fix this inline for now.
                         cv2.imwrite(str(Path("collected_dataset/full_images") / full_name), img)
                         cv2.imwrite(str(Path("collected_dataset/crops") / crop_name), crop_img)
                         
                    db.increment_stat('total_uploads')
                    st.session_state.processed_files.add(file_id)
                    st.success(f"✅ Saved {len(results)} plates!")
                    st.rerun() # Refresh KPIs

# ---------------------------------------------------------------------
# TAB 2: Video
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### 🎬 Video Analysis")
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if uploaded_video and st.button("Start Video Analysis", use_container_width=True):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        bar = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
                frame = apply_enhancements(frame)
                results = reader.predict(frame, conf_thres=conf_thres)
                
                if results:
                    bid = str(uuid.uuid4())[:8]
                    for res in results:
                        x1, y1, x2, y2 = res['box']
                        crop_img = frame[y1:y2, x1:x2]
                        
                        fname = f"vid_{bid}_{count}.jpg"
                        cname = f"crop_{bid}_{count}.jpg"
                        
                        cv2.imwrite(str(Path("collected_dataset/full_images") / fname), frame)
                        cv2.imwrite(str(Path("collected_dataset/crops") / cname), crop_img)
                        
                        db.add_entry(fname, cname, res['text'], res['conf'], bid)
                        count += 1
                        
            cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if total > 0: bar.progress(min(cur_pos/total, 1.0))
        
        cap.release()
        st.success(f"✅ Extracted {count} plates!")
        st.rerun()

# ---------------------------------------------------------------------
# TAB 3: Live
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### 🔴 Cyber-HUD Live")
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = reader.predict(img, conf_thres=conf_thres)
        annotated = reader.visualize(img, results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_frame_callback=video_frame_callback)

# ---------------------------------------------------------------------
# TAB 4: IP Cams (Scanner Integrated)
# ---------------------------------------------------------------------
with tab4:
    st.markdown("### 📡 IP Camera Hub")
    
    col_scan, col_view = st.columns([1, 2])
    
    with col_scan:
        st.warning("⚠️ Scanner is intensive. Run only if necessary.")
        if st.button("Start Network Scan"):
            scan_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            def update_progress(p):
                progress_bar.progress(p)
            
            results = network_scanner.scan_network(progress_callback=update_progress)
            
            if not results:
                st.error("No Cameras Found")
            else:
                st.success(f"Found {len(results)} Devices")
                for dev in results:
                    st.json(dev)
                    urls = network_scanner.get_rtsp_urls(dev)
                    for url in urls:
                         st.code(url, language="text")
    
    with col_view:
        rtsp_url = st.text_input("RTSP Stream URL", "rtsp://admin:12345@192.168.1.64:554/stream")
        if st.checkbox("Start Stream"):
            st_img = st.empty()
            cap = cv2.VideoCapture(rtsp_url)
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Connection Failed")
                    break
                
                results = reader.predict(frame, conf_thres=conf_thres)
                viz = reader.visualize(frame, results)
                st_img.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True)

# ---------------------------------------------------------------------
# TAB 5: ADMIN (Refined)
# ---------------------------------------------------------------------
with tab5:
    if auth.check_password():
        st.success("Authorized Access")
        st.markdown("---")
        
        # Tinder-style Review? Or Data Editor?
        # Let's stick to Data Editor but backed by SQLite
        
        df = db.get_all()
        if not df.empty:
            st.markdown("### 📝 Dataset Management")
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "crop_image_path": st.column_config.ImageColumn("Crop", width=100), # Note: needs valid URL/Path. Streamlit ImageColumn tricky with local files.
                    "review_status": st.column_config.SelectboxColumn("Status", options=["PENDING", "VERIFIED", "WRONG"])
                },
                disabled=["id", "timestamp"],
                use_container_width=True,
                key="editor"
            )
            
            if st.button("💾 Commit Changes"):
                # Detect changes (Naive approach: Loop all?)
                # SQLite update is fast. 
                # Ideally we only update changed rows.
                # Streamlit data_editor returns all data.
                
                # For DB: Iterate over edited_df and update DB
                # This is heavy for large DBs but fine for <1000 rows
                count = 0 
                for index, row in edited_df.iterrows():
                    # Only update if changed? 
                    # We just blind update for simplicity or check diff
                     db.update_status(row['id'], row['review_status'], row['predicted_text'])
                     count+=1
                st.success(f"Updated {count} records.")
        else:
            st.info("Database empty.")
