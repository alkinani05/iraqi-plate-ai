
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
    page_title="Algonest AI | Scanner", 
    page_icon="👁️", 
    layout="centered", # Centered is better for mobile-first feel
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------
# 🎨 STAR-SYSTEM DESIGN (CSS)
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@400;700&display=swap');
    
    body {
        background-color: #000;
        color: #fff;
    }
    
    .stApp {
        background: #000; /* Pure Black for OLED */
    }

    /* 🔥 SCANNER MODE TOGGLE */
    .stRadio > div {
        flex-direction: row;
        justify-content: center;
        background: #111;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #333;
    }
    
    div[data-baseweb="radio"] > div {
        background: transparent;
        border-radius: 8px;
        color: #888;
        padding: 5px 15px;
        font-family: 'Outfit', sans-serif;
    }
    
    div[aria-checked="true"] {
        background-color: #00ff80 !important; /* Cyber Green */
        color: #000 !important;
        font-weight: bold;
        box-shadow: 0 0 10px rgba(0,255,128,0.5);
    }
    
    h1 {
        font-family: 'Share Tech Mono', monospace;
        color: #00ff80;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1.8rem;
        margin-bottom: 0px;
        text-shadow: 0 0 10px rgba(0,255,128,0.4);
    }
    
    .status-bar {
        font-family: 'Share Tech Mono', monospace;
        display: flex;
        justify-content: space-between;
        padding: 5px 10px;
        background: #0a0a0a;
        border-bottom: 1px solid #333;
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 20px;
    }

    /* 📸 VIEWFINDER STYLE */
    video {
        border-radius: 4px !important;
        border: 2px solid #333 !important;
        box-shadow: 0 0 30px rgba(0,255,128,0.1);
        width: 100% !important;
        max-height: 80vh;
    }

    /* 🧠 Image Result Optimization */
    .stImage > img {
        border-radius: 8px;
        border: 1px solid #333;
    }
    
    /* Hide Junk */
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

# ---------------------------------------------------------------------
# 📹 VIDEO PROCESSOR
# ---------------------------------------------------------------------
class PlateVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.reader = load_model()
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_results = []
        self.fps = 0
        self.prev_time = 0

    def recv(self, frame):
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        
        # FPS Calc
        if self.prev_time > 0:
            dt = current_time - self.prev_time
            if dt > 0: self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = current_time

        # Mobile Optimization
        height, width = img.shape[:2]
        target_width = 800
        scale = 1.0
        
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            img_small = cv2.resize(img, (target_width, new_height))
        else:
            img_small = img

        self.frame_count += 1
        run_ai = (self.frame_count % 3 == 0)
        results = self.last_results
        
        if run_ai:
            with self.lock:
                small_results = self.reader.predict(img_small)
                results = []
                for res in small_results:
                    x1, y1, x2, y2 = res['box']
                    results.append({
                        'box': [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)],
                        'text': res['text'],
                        'conf': res['conf']
                    })
                self.last_results = results
        
        annotated_img = self.reader.visualize(img, results, fps=self.fps)
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# ---------------------------------------------------------------------
# 📱 UI LAYOUT
# ---------------------------------------------------------------------
st.markdown("""
    <div class="status-bar">
        <span>SYSTEM: ONLINE</span>
        <span>NET: SECURE</span>
        <span>VER: 3.4 STABLE</span>
    </div>
    <h1>ALGONEST | SECURITY</h1>
""", unsafe_allow_html=True)

# Advanced Mode Toggle
mode = st.radio("OPERATIONAL MODE", ["SCANNER", "ANALYSIS"], horizontal=True, label_visibility="collapsed")

if mode == "SCANNER":
    st.markdown("<div style='text-align:center; color:#666; font-size:0.8rem; margin-top:5px; margin-bottom:10px;'>Align vehicle plate within the frame</div>", unsafe_allow_html=True)
    
    # 🔥 STABILITY FIX: REDUCED POOL SIZE
    # The 'NoneType has no attribute sendto' error happens when
    # the socket closes while STUN is still retrying.
    # We reduce pool size to 1 to prevent socket saturation.
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
        ],
        "iceCandidatePoolSize": 0, # DISABLED PRE-FETCH to prevent race condition
        "iceTransportPolicy": "all",
    })
    
    # 🛡️ SAFE STREAMER WRAPPER
    try:
        webrtc_ctx = webrtc_streamer(
            key="plate-scanner-stable",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "audio": False,
                "video": {
                    "facingMode": "environment",
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30, "max": 60}
                }
            },
            video_processor_factory=PlateVideoTransformer,
            async_processing=True,
        )
    except Exception as e:
        st.error(f"Stream Error: {e}")
        webrtc_ctx = None

elif mode == "ANALYSIS":
    st.markdown("### 🖼️ STATIC ANALYSIS")
    uploaded = st.file_uploader("Upload Surveillance Image", type=['jpg','png','jpeg'])
    
    if uploaded:
        col1, col2, col3 = st.columns([1, 6, 1]) # Column layout to center and control size
        with col2:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("DECRYPTING VISUAL DATA..."):
                results = reader.predict(img)
                viz = reader.visualize(img, results)
                
                # Dynamic sizing based on aspect ratio
                st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True, caption=f"Processing Complete: {len(results)} Targets Found")
                
                if results:
                    st.success(f"TARGET ACQUIRED: {results[0]['text']}")

