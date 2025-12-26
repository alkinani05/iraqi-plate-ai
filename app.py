
import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_plate_reader import PlateReader
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# ---------------------------------------------------------------------
# ⚙️ CONFIG & PAGE SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Algonest AI | Plate Reader", 
    page_icon="🦅", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------
# 🎨 PLATINUM DESIGN SYSTEM (CSS)
# ---------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Reset & Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #050a10; 
        color: #e6edf3;
    }

    /* 💎 Ultimate Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1f2937 0%, #0d1117 40%, #000000 100%);
        background-attachment: fixed;
    }
    
    /* 🌟 Header Styling */
    h1 {
        background: linear-gradient(90deg, #ffd700 0%, #ffffff 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem !important;
        text-align: center;
        margin-bottom: 0rem !important;
        letter-spacing: -1px;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    }
    
    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 0.9rem;
        letter-spacing: 4px;
        margin-bottom: 2rem;
        text-transform: uppercase;
        opacity: 0.8;
    }

    /* 🔘 Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255,255,255,0.05);
        padding: 5px;
        border-radius: 50px;
        display: flex;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 40px;
        color: #8b949e;
        flex: 1;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffd700 !important;
        color: #000 !important;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
    }

    /* 📸 Video Container */
    video {
        border-radius: 12px !important;
        border: 1px solid #333 !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
        width: 100% !important;
    }
    
    /* Center the WebRTC component */
    div[data-testid="stVerticalBlock"] > div > div > div > div > video {
        margin: 0 auto;
        display: block;
    }

    /* ✨ Result Cards */
    div.stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #e6edf3;
        border-radius: 10px;
        text-align: center;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(10px);
        padding: 10px;
        text-align: center;
        font-size: 0.7rem;
        color: #666;
        border-top: 1px solid #222;
        z-index: 999;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 🧠 AI LOGIC
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return PlateReader()

# Load early to fail fast if issues
try:
    reader = load_model()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# 📹 VIDEO PROCESSOR
# ---------------------------------------------------------------------
class PlateVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.reader = load_model()
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_results = []
        
        # FPS Tracking
        self.prev_time = 0
        self.fps = 0
        self.infer_ms = 0

    def recv(self, frame):
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        
        # FPS Calc
        if self.prev_time > 0:
            dt = current_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = current_time

        # 1. Resize for Mobile Performance
        height, width = img.shape[:2]
        target_width = 800
        scale = 1.0
        
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            img_small = cv2.resize(img, (target_width, new_height))
        else:
            img_small = img

        # 2. Skip Frames for Inference Speed
        self.frame_count += 1
        run_ai = (self.frame_count % 3 == 0) # Run every 3rd frame
        results = self.last_results
        
        if run_ai:
            with self.lock:
                t0 = time.time()
                small_results = self.reader.predict(img_small)
                self.infer_ms = (time.time() - t0) * 1000
                
                # Descale coords
                results = []
                for res in small_results:
                    x1, y1, x2, y2 = res['box']
                    results.append({
                        'box': [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)],
                        'text': res['text'],
                        'conf': res['conf']
                    })
                self.last_results = results
        
        # 3. Visualize
        try:
            annotated_img = self.reader.visualize(img, results, fps=self.fps, inference_ms=self.infer_ms)
        except Exception as e:
            annotated_img = img # Fallback
            
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# ---------------------------------------------------------------------
# 📱 UI LAYOUT
# ---------------------------------------------------------------------
st.markdown("<h1>ALGONEST AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Professional License Plate Recognition System v3.0</p>", unsafe_allow_html=True)

# Main Interface
tab_cam, tab_up = st.tabs(["🎥 LIVE SCANNER", "🖼️ PHOTO ANALYZER"])

with tab_cam:
    
    st.markdown("### 🔴 Real-Time Secure Stream")
    
    # STUN Config for firewall traversal
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Mobile-First Media Constraints
    CONSTRAINTS = {
        "audio": False,
        "video": {
            "facingMode": "environment", # Prefer back camera
            "width": {"min": 640, "ideal": 1280, "max": 1920},
            "height": {"min": 480, "ideal": 720, "max": 1080},
        }
    }
    
    webrtc_ctx = webrtc_streamer(
        key="plate-scanner-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints=CONSTRAINTS,
        video_processor_factory=PlateVideoTransformer,
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.caption("⚡ Powered by Algonest Neural Engine")
    else:
        st.info("👆 Tap START to activate camera system")

with tab_up:
    st.markdown("### 🧬 Static Analysis")
    uploaded = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
    
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Analyze
        with st.spinner("Processing..."):
            results = reader.predict(img)
            viz = reader.visualize(img, results)
            
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            if results:
                for r in results:
                    st.success(f"DETECTED: **{r['text']}** (Confidence: {int(r['conf']*100)}%)")
            else:
                st.warning("No plates detected in this image.")

st.markdown("""
<div class='footer'>
    Algonest Artificial Intelligence • Baghdad, Iraq • 2025<br>
    System Status: 🟢 Operational
</div>
""", unsafe_allow_html=True)
