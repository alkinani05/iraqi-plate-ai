
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
from pathlib import Path

# OPTIMIZATION
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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
# SMART GARAGE PLATE READER CLASS (FANTASTIC EDITION)
# ------------------------------------
class PlateReader:
    def __init__(self, detector_path=None, ocr_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Robust Path Resolution
        # Current file: src/core/plate_reader.py
        # Models: models/
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        self.models_dir = project_root / "models"
        
        if detector_path is None: detector_path = self.models_dir / "detector.pt"
        if ocr_path is None: ocr_path = self.models_dir / "professional_real_only.pth"

        print(f"🚀 [INIT] Device: {self.device}")
        
        # 2. Load Models
        if detector_path.exists():
             self.detector = YOLO(str(detector_path))
             print(f"✅ Detector Loaded: {detector_path.name}")
        else:
             print(f"❌ Critical: Detector missing at {detector_path}")

        # 3. Load OCR
        LAT_DIGITS = "0123456789"
        LAT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
        # Common Iraqi Plate Characters
        AR_LETTERS = "أبجدهوزحطيكمنصعفقشترثخذضظغ" 
        ALL_CHARS = sorted(list(set(LAT_DIGITS + LAT_LETTERS + AR_DIGITS + AR_LETTERS)))
        self.chars = "".join(ALL_CHARS)
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        
        try:
            self.ocr_model = CRNN(len(self.chars))
            if ocr_path.exists():
                self.ocr_model.load_state_dict(torch.load(str(ocr_path), map_location=self.device))
                print(f"✅ OCR Model Loaded: {ocr_path.name}")
            self.ocr_model.to(self.device).eval()
        except Exception as e:
            print(f"❌ Error loading OCR: {e}")

        # 4. Transforms
        self.img_h, self.img_w = 64, 256
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        self._load_fonts()

    def _load_fonts(self):
        fonts_dir = self.models_dir / "fonts"
        
        def load_font(name, size):
            p = fonts_dir / name
            return lambda s: ImageFont.truetype(str(p), int(s)) if p.exists() else ImageFont.load_default()

        # Dynamic Sizing based on Scale 'k'
        self.get_font_ar = load_font("NotoSansArabic-Bold.ttf", 40)
        self.get_font_en = load_font("NotoSans-Bold.ttf", 40)

    def decode_text(self, preds):
        preds = preds.squeeze(1).argmax(1).cpu().numpy()
        decoded = []
        prev = 0
        for p in preds:
            if p != 0 and p != prev:
                decoded.append(self.idx2char.get(p, ""))
            prev = p
        return "".join(decoded)

    def predict(self, image_path_or_array, conf_thres=0.35):
        if isinstance(image_path_or_array, str): img0 = cv2.imread(image_path_or_array)
        else: img0 = image_path_or_array
        if img0 is None or not hasattr(self, 'detector'): return []

        results = self.detector.predict(img0, conf=conf_thres, verbose=False)
        output = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                h, w = crop.shape[:2]
                if w < h or w < 20 or h < 10: continue

                # OCR Inference
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    preds = self.ocr_model(tensor)
                    text = self.decode_text(preds)
                
                if len(text) > 0:
                    text = self._post_process(text)
                    output.append({'box': [x1, y1, x2, y2], 'text': text, 'conf': conf})
        return output

    def _post_process(self, text):
        """
        Iraqi Plate Rules v2.1 (FIXED)
        """
        text = text.strip().replace(" ", "")
        
        # 1. Clean Ghosts
        # Leading/Trailing symbols that aren't digits/letters
        cleaned = ""
        for char in text:
            if char.isalnum(): cleaned += char
        text = cleaned

        # 2. Character Replacement (Fix applied)
        replacements = {
            'O': '0', 'o': '0',
            'I': '1', 'i': '1', 'l': '1',
            'Z': '2', 'z': '2',
            'B': '8', 
            'S': '5', 's': '5',
            'G': '6', 'g': '6',
            'A': '4' 
        }
        
        # Apply replacements
        new_text = list(text)
        for i, char in enumerate(new_text):
            if char in replacements:
                new_text[i] = replacements[char]
        
        text = "".join(new_text)

        # 3. Structural Logic
        if len(text) >= 2:
             if text[0] == 'L': text = '1' + text[1:]
             
        return text

    def visualize(self, img, results, fps=None):
        """
        CYBER-HUD v2.0
        """
        annotated_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_pil, 'RGBA')
        w, h = annotated_pil.size
        # Dynamic Scaling
        k = max(h / 1000, 0.6) 
        
        # Colors
        c_cyan = (0, 255, 255, 255)
        c_green = (0, 255, 128, 255)
        c_red = (255, 50, 50, 255)
        c_bg_dark = (0, 0, 0, 180)

        f_lg = self.get_font_en(int(45*k))
        f_ar = self.get_font_ar(int(40*k))
        f_sm = self.get_font_en(int(20*k))

        # 1. Grid Effect (Subtle)
        step = int(100 * k)
        for i in range(0, w, step):
            draw.line([(i, 0), (i, h)], fill=(0, 255, 255, 10), width=1)
        for i in range(0, h, step):
            draw.line([(0, i), (w, i)], fill=(0, 255, 255, 10), width=1)

        # 2. Scope Corners
        m = 20
        l = 40
        for x, y, dx, dy in [(m,m,1,1), (w-m,m,-1,1), (m,h-m,1,-1), (w-m,h-m,-1,-1)]:
            draw.line([(x,y), (x+dx*l, y)], fill=c_cyan, width=3)
            draw.line([(x,y), (x, y+dy*l)], fill=c_cyan, width=3)

        # 3. Detections
        detected_text = None
        
        for res in results:
            x1, y1, x2, y2 = res['box']
            text = res['text']
            conf = res['conf']
            
            color = c_green if conf > 0.6 else c_cyan
            
            # Glowing Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Corner accents
            d = 10
            draw.line([(x1, y1), (x1+d, y1)], fill=(255,255,255,255), width=5)
            draw.line([(x2, y2), (x2-d, y2)], fill=(255,255,255,255), width=5)

            # Label Tag
            label = f"{text} ({int(conf*100)}%)"
            # Draw Arabic safe text
            if any('\u0600' <= c <= '\u06FF' for c in text):
                disp = get_display(arabic_reshaper.reshape(text))
                font = f_ar
            else:
                disp = text
                font = f_lg
            
            # Label Background
            tw = draw.textlength(disp, font=font)
            draw.rectangle([x1, y1 - 50*k, x1 + tw + 20, y1], fill=c_bg_dark)
            draw.text((x1+10, y1 - 50*k), disp, font=font, fill=color)

            detected_text = text # Capture for HUD

        # 4. HUD Bottom Bar
        if detected_text:
            dh = int(80*k)
            draw.rectangle([0, h-dh, w, h], fill=(0,10,20,240))
            draw.line([0, h-dh, w, h-dh], fill=c_green, width=2)
            draw.text((20, h-dh+10), f"TARGET: {detected_text}", font=f_lg, fill=c_green)
            draw.text((w-200, h-dh+10), "ALGONEST AI", font=f_sm, fill=(255,255,255,100))
        
        return cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGBA2BGR)
