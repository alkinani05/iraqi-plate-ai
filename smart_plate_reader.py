
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
from torchvision import transforms

# ------------------------------------
# CORE ARCHITECTURE
# ------------------------------------
class CRNN(nn.Module):
    """
    Compact CRNN Architecture for License Plate Recognition.
    """
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.lstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_chars + 1)

    def forward(self, x):
        features = self.cnn(x).mean(2).permute(0, 2, 1)
        return self.fc(self.lstm(features)[0]).permute(1, 0, 2)

# ------------------------------------
# SMART GARAGE PLATE READER CLASS
# ------------------------------------
class PlateReader:
    def __init__(self, detector_path=None, ocr_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Path Resolution (Priority: Project Root -> Local Fallback)
        if detector_path is None:
            detector_path = os.path.join(self.base_dir, "detector.pt")
            
        if ocr_path is None:
            # Models to check in order of preference
            candidates = [
                "professional_real_only.pth",
                "ocr_model.pth"
            ]
            for cand in candidates:
                p = os.path.join(self.base_dir, cand)
                if os.path.exists(p):
                    ocr_path = p
                    break
            
            # If still None, checking development paths
            if ocr_path is None:
                 print("⚠️ Model not found in root, checking dev paths...")
                 ocr_path = os.path.join(self.base_dir, "professional_real_only.pth") # Default fallback

        print(f"🚀 [INIT] Device: {self.device}")
        
        # 2. Load Detector
        if not os.path.exists(detector_path):
             print(f"❌ Critical: Detector not found at {detector_path}")
        else:
             self.detector = YOLO(detector_path)
             print(f"✅ Detector Loaded: {os.path.basename(detector_path)}")
        
        # 3. Load OCR
        # Universal Vocab (Professional)
        LAT_DIGITS = "0123456789"
        LAT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
        AR_LETTERS = "أبجدهوزحطيكمنصعفقشترثخذضظغ"
        ALL_CHARS = sorted(list(set(LAT_DIGITS + LAT_LETTERS + AR_DIGITS + AR_LETTERS)))
        self.chars = "".join(ALL_CHARS)
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        
        try:
            self.ocr_model = CRNN(len(self.chars))
            if ocr_path and os.path.exists(ocr_path):
                self.ocr_model.load_state_dict(torch.load(ocr_path, map_location=self.device))
                print(f"✅ OCR Model Loaded: {os.path.basename(ocr_path)}")
            else:
                print("⚠️ System Notice: Running in Uninitialized State (Model Weights Pending).")
            
            self.ocr_model.to(self.device)
            self.ocr_model.eval()
        except Exception as e:
            print(f"❌ Error loading OCR model: {e}")

        # 4. Transforms
        self.img_h, self.img_w = 64, 256
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        # 5. Fonts
        self._load_fonts()

    def _load_fonts(self):
        fonts_dir = os.path.join(self.base_dir, "fonts")
        
        def load_font(name, default_size=20):
            path = os.path.join(fonts_dir, name)
            if os.path.exists(path):
                return lambda s: ImageFont.truetype(path, s)
            return lambda s: ImageFont.load_default()

        self.get_font_ar = load_font("NotoSansArabic-Bold.ttf")
        self.get_font_en = load_font("NotoSans-Bold.ttf")

    def decode_text(self, preds):
        preds = preds.squeeze(1).argmax(1).cpu().numpy()
        decoded = []
        prev = 0
        for p in preds:
            if p != 0 and p != prev:
                decoded.append(self.idx2char.get(p, ""))
            prev = p
        return "".join(decoded)

    def predict(self, image_path_or_array):
        """
        Takes an image path or numpy array (cv2 image).
        Returns list of dicts: [{'box': [x1,y1,x2,y2], 'text': '12345', 'conf': 0.95}, ...]
        """
        if isinstance(image_path_or_array, str):
            img0 = cv2.imread(image_path_or_array)
        else:
            img0 = image_path_or_array
            
        if img0 is None: return []
        
        if not hasattr(self, 'detector'):
            return []

        # Detect
        # Lower confidence slightly for better recall on mobile streams
        results = self.detector.predict(img0, conf=0.35, verbose=False)
        output = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Check aspect ratio to avoid noise
                h, w = crop.shape[:2]
                if w < h or w < 20 or h < 10: continue

                # OCR
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    preds = self.ocr_model(tensor)
                    text = self.decode_text(preds)
                
                # Basic Noise Filter (must contain at least 1 digit)
                if len(text) > 0:
                    text = self._post_process(text)
                    output.append({
                        'box': [x1, y1, x2, y2],
                        'text': text,
                        'conf': conf
                    })
        return output

    def _post_process(self, text):
        """
        Apply Iraqi Plate Heuristics
        """
        text = text.strip().replace(" ", "")
        
        # Rule 1: 9-digit noise handling (Leading 1 or 4 often ghost artifact)
        if len(text) == 9 and text[0] in ['1', '4', '7'] and text[1:].isdigit():
             candidate = text[1:] 
             # Check for Letter pattern in what remains (NN L NNNNN)
             if candidate[:2].isdigit() and candidate[3:].isdigit():
                  suspicious_char = candidate[2]
                  if suspicious_char in ['1', '4', '3', '7']: 
                       return candidate[:2] + "A" + candidate[3:]
             return candidate

        # Rule 2: 8-character pattern mapping (NN L NNNNN)
        if len(text) == 8 and text[:2].isdigit() and text[3:].isdigit():
             if text[2] in ['1', '4', '7']: 
                 return text[:2] + "A" + text[3:]
                 
        return text

    def visualize(self, img, results, fps=None, inference_ms=None):
        """
        ADVANCED SCANNER HUD: Professional Military/Sci-Fi Aesthetic
        """
        # Helper function
        def to_english_digits(text):
            eastern_to_western = {
                '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
                '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
            }
            return "".join([eastern_to_western.get(c, c) for c in text])

        annotated_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_pil, 'RGBA')
        w, h = annotated_pil.size
        
        # Scale factor (Optimized for Mobile)
        k = max(h / 1000, 0.6) 
        
        # PALETTE: Clean White/Red/Green (Professional Security Look)
        # Instead of generic neon, we use high-contrast "OS" colors.
        col_scan = (255, 255, 255, 150)    # White dim
        col_lock = (0, 255, 128, 255)      # Spring Green (Lock)
        col_alert = (255, 50, 50, 255)     # Red
        col_bg = (10, 10, 10, 220)         # Almost Black
        
        # Dynamic Fonts
        f_xl = self.get_font_en(int(55 * k))
        f_ar = self.get_font_ar(int(45 * k))
        f_md = self.get_font_en(int(22 * k))
        f_sm = self.get_font_en(int(16 * k))
        f_xs = self.get_font_en(int(12 * k))

        # -----------------------------------------------
        # 1. CENTRAL RETICLE (Always Visible)
        # -----------------------------------------------
        cx, cy = w//2, h//2
        ret_len = int(30 * k)
        gap = int(10 * k)
        
        # Crosshair center
        draw.line([(cx-ret_len, cy), (cx-gap, cy)], fill=col_scan, width=1)
        draw.line([(cx+gap, cy), (cx+ret_len, cy)], fill=col_scan, width=1)
        draw.line([(cx, cy-ret_len), (cx, cy-gap)], fill=col_scan, width=1)
        draw.line([(cx, cy+gap), (cx, cy+ret_len)], fill=col_scan, width=1)
        
        # Corner Brackets (Scope)
        scope_w, scope_h = int(w*0.8), int(h*0.6)
        sx1, sy1 = (w - scope_w)//2, (h - scope_h)//2
        sx2, sy2 = sx1 + scope_w, sy1 + scope_h
        slen = int(40 * k)
        
        # Draw Scope Corners
        for (px, py, dx, dy) in [(sx1, sy1, 1, 1), (sx2, sy1, -1, 1), (sx1, sy2, 1, -1), (sx2, sy2, -1, -1)]:
            draw.line([(px, py), (px + dx*slen, py)], fill=col_scan, width=2)
            draw.line([(px, py), (px, py + dy*slen)], fill=col_scan, width=2)

        # -----------------------------------------------
        # 2. DETECTION LOGIC
        # -----------------------------------------------
        detected_text = None
        best_conf = 0
        
        for res in results:
            x1, y1, x2, y2 = res['box']
            text = to_english_digits(res['text'])
            conf = res.get('conf', 0)
            
            if conf > best_conf:
                best_conf = conf
                detected_text = text

            # Lock-on Bracket
            is_verified = conf > 0.45
            color = col_lock if is_verified else col_scan
            
            # Animate Bracket Expansion
            m = 5 # margin
            
            # Draw Dynamic Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Label
            if is_verified:
                tag = f"CONF: {int(conf*100)}%"
                draw.rectangle([x1, y1-30*k, x1+120*k, y1], fill=color)
                draw.text((x1+5, y1-25*k), tag, font=f_sm, fill=(0,0,0,255))

        # -----------------------------------------------
        # 3. ADVANCED RESULTS PANE (Bottom)
        # -----------------------------------------------
        # Only show if we strongly detect something
        if detected_text and best_conf > 0.45:
            # Reshape Arabic
            if any('\u0600' <= c <= '\u06FF' for c in detected_text):
                disp_text = get_display(arabic_reshaper.reshape(detected_text))
                font_main = f_ar
            else:
                disp_text = detected_text
                font_main = f_xl

            # Panel Geometry
            pan_h = int(140 * k)
            draw.rectangle([0, h-pan_h, w, h], fill=col_bg)
            draw.line([0, h-pan_h, w, h-pan_h], fill=col_lock, width=2)
            
            # Side Decor
            draw.rectangle([0, h-pan_h, 10, h], fill=col_lock) # Left strip
            
            # Text Rendering
            draw.text((30, h-pan_h+15), "TARGET IDENTIFIED", font=f_sm, fill=col_lock)
            draw.text((30, h-pan_h+40), disp_text, font=font_main, fill=(255,255,255,255))
            
            # Meta info
            draw.text((w-150*k, h-pan_h+20), f"ACCURACY", font=f_xs, fill=(150,150,150,255))
            draw.text((w-150*k, h-pan_h+40), f"{int(best_conf*100)}%", font=f_xl, fill=col_lock)
            
        else:
            # System Idle / Scanning
            draw.text((cx, h - int(50*k)), "• SYSTEM ACTIVE •", font=f_md, anchor="mm", fill=(200,200,200,200))

        # -----------------------------------------------
        # 4. TOP TELEMETRY
        # -----------------------------------------------
        draw.rectangle([0, 0, w, 40*k], fill=(0,0,0,150))
        draw.text((15, 8*k), "REC ●", font=f_sm, fill=col_alert)
        if fps:
            draw.text((w-100*k, 8*k), f"{int(fps)} FPS", font=f_sm, fill=(0,255,0,200))
        
        return cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGBA2BGR)
