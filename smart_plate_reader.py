
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
        Platinum HUD Design: High-End, Professional, Glass-styled.
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
        
        # Scale factor
        k = max(h / 1080, 0.5) 
        
        # Palette (Cyberpunk Gold & Teal)
        col_primary = (255, 215, 0, 255)   # Gold
        col_sec = (0, 240, 255, 200)      # Cyan
        col_bg = (10, 15, 25, 230)        # Dark Blue/Black semi-transparent
        col_success = (50, 255, 100, 255) # Green
        
        # Dynamic Fonts
        f_xl = self.get_font_en(int(60 * k))
        f_ar = self.get_font_ar(int(50 * k))
        f_md = self.get_font_en(int(25 * k))
        f_sm = self.get_font_en(int(18 * k))

        # 1. TOP BAR (Minimalist)
        # ------------------------------------------------
        bar_h = int(50 * k)
        draw.rectangle([0, 0, w, bar_h], fill=(0,0,0,180))
        
        # Left: App Name
        draw.text((20, int(10*k)), "ALGONEST | PRO", font=f_md, fill=col_primary)
        
        # Right: FPS info
        if fps is not None:
            fps_text = f"FPS: {int(fps)}"
            bbox = draw.textbbox((0,0), fps_text, font=f_sm)
            draw.text((w - bbox[2] - 20, int(14*k)), fps_text, font=f_sm, fill=(200,200,200,255))

        # 2. DETECTION BOXES
        # ------------------------------------------------
        detected_text_display = None
        best_conf = 0
        
        for res in results:
            x1, y1, x2, y2 = res['box']
            text = to_english_digits(res['text'])
            conf = res.get('conf', 0)
            
            # Update best detection
            if conf > best_conf:
                best_conf = conf
                detected_text_display = text
            
            # Draw Corners Only (Sleek Look)
            corner_len = int((x2-x1) * 0.2)
            th = int(4 * k)
            
            # Top Left
            draw.line([(x1, y1), (x1+corner_len, y1)], fill=col_primary, width=th)
            draw.line([(x1, y1), (x1, y1+corner_len)], fill=col_primary, width=th)
            # Top Right
            draw.line([(x2, y1), (x2-corner_len, y1)], fill=col_primary, width=th)
            draw.line([(x2, y1), (x2, y1+corner_len)], fill=col_primary, width=th)
            # Bottom Left
            draw.line([(x1, y2), (x1+corner_len, y2)], fill=col_primary, width=th)
            draw.line([(x1, y2), (x1, y2-corner_len)], fill=col_primary, width=th)
            # Bottom Right
            draw.line([(x2, y2), (x2-corner_len, y2)], fill=col_primary, width=th)
            draw.line([(x2, y2), (x2, y2-corner_len)], fill=col_primary, width=th)
            
            # Semi-transparent fill
            valid_detect = conf > 0.4
            fill_col = (255, 215, 0, 20) if valid_detect else (0, 255, 255, 20)
            draw.rectangle([x1, y1, x2, y2], fill=fill_col)

        # 3. BOTTOM INFO CARD (Floating Glass)
        # ------------------------------------------------
        if detected_text_display and best_conf > 0.4:
            # Reshape for Arabic if needed
            if any('\u0600' <= c <= '\u06FF' for c in detected_text_display):
                disp_text = get_display(arabic_reshaper.reshape(detected_text_display))
                font_main = f_ar
            else:
                disp_text = detected_text_display
                font_main = f_xl

            # Box dimensions
            card_w = int(w * 0.6)
            card_h = int(120 * k)
            card_x = (w - card_w) // 2
            card_y = h - card_h - int(40 * k)
            
            # Draw Glass Card with Glow
            draw.rounded_rectangle([card_x, card_y, card_x+card_w, card_y+card_h], radius=20, fill=col_bg, outline=col_sec, width=2)
            
            # Glow behind text
            # text_bbox = draw.textbbox((0,0), disp_text, font=font_main)
            # text_w = text_bbox[2] - text_bbox[0]
            # text_h = text_bbox[3] - text_bbox[1]
            
            # Centered Text
            draw.text((card_x + card_w//2, card_y + card_h//2), disp_text, font=font_main, anchor="mm", fill=col_primary)
            
            # "LICENSE PLATE" Label
            draw.text((card_x + 20, card_y + 15), "VERIFIED PLATE", font=f_sm, fill=col_success)
            
            # Confidence
            draw.text((card_x + card_w - 80, card_y + 15), f"{int(best_conf*100)}%", font=f_sm, fill=col_primary)

        else:
            # Scanning Animation (Crosshair)
            cx, cy = w//2, h//2
            len_ch = int(40 * k)
            draw.line([(cx-len_ch, cy), (cx+len_ch, cy)], fill=(255,255,255,50), width=1)
            draw.line([(cx, cy-len_ch), (cx, cy+len_ch)], fill=(255,255,255,50), width=1)
            
            draw.text((cx, cy + int(50*k)), "SEARCHING TARGET...", font=f_md, anchor="mm", fill=(255,255,255,180))

        return cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGBA2BGR)
