# app.py
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import easyocr
import os
import json
import re

# Config Yükle
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

config = load_config()

# Gemini API Kurulumu
GEMINI_ENABLED = False
gemini_model = None

try:
    import google.generativeai as genai
    GEMINI_API_KEY = config.get('GEMINI_API_KEY', '')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        GEMINI_ENABLED = True
        print("✅ Gemini API aktif (Hakem Modu)")
    else:
        print("⚠️ API key yok")
except Exception as e:
    print(f"⚠️ Gemini yüklenemedi: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# ---------------- MODEL YÜKLEME ----------------
print("Model yükleniyor...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load("ekg_models_resnet50/resnet50_ptbxl_gray_fold3.pth", map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print("✅ ResNet50 Model Yüklendi")

print("OCR yükleniyor...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR Yüklendi")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- YARDIMCI FONKSİYONLAR ----------------

def check_if_ecg(img_path):
    """Görüntü EKG mi kontrolü"""
    if not GEMINI_ENABLED:
        return True, ""
    try:
        img = Image.open(img_path)
        prompt = "Bu görüntü bir EKG mi? Sadece 'EVET' veya 'HAYIR' yaz."
        response = gemini_model.generate_content([prompt, img])
        return ("HAYIR" not in (response.text or "").upper()), (response.text or "")
    except:
        return True, ""

def verify_with_gemini(img_path, model_prediction, model_confidence):
    """
    Gemini HAKEM:
    - model_dogru_mu = true  -> model_prediction aynen kalır
    - model_dogru_mu = false -> final_tani ile sadece o zaman düzeltir
    Stabil olsun diye SADECE JSON ister ve JSON'u regex ile çeker.
    """
    if not GEMINI_ENABLED or gemini_model is None:
        return {
            "final_sonuc": model_prediction,
            "duzeltme_yapildi": False,
            "yorum": "AI API kapalı, model sonucu baz alındı."
        }

    try:
        img = Image.open(img_path)

        prompt = f"""
Sen uzman bir kardiyologsun. EKG görüntüsünü incele ve modelin ön tahminini değerlendir.

MODEL_TAHMINI: {model_prediction}
MODEL_GUVEN: %{model_confidence}

SADECE geçerli JSON döndür. JSON dışında TEK KARAKTER yazma.
Şema birebir şu olacak:
{{
  "model_dogru_mu": true,
  "final_tani": "NORMAL",
  "klinik_yorum": "2-3 cümle, sakin ve yönlendirici Türkçe yorum. Kesin tanı koyma, kardiyolog kontrolü öner."
}}

Kurallar:
- final_tani sadece "NORMAL" veya "ABNORMAL" olabilir.
- model_dogru_mu sadece true/false olabilir.
- klinik_yorum 2-3 cümle, tek paragraf.
""".strip()

        resp = gemini_model.generate_content([prompt, img])
        raw = (resp.text or "").strip()

        # JSON gövdesini yakala (```json ... ``` olsa bile)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise ValueError("Gemini JSON döndürmedi")

        data = json.loads(m.group(0))

        model_ok = bool(data.get("model_dogru_mu", True))
        final_tani = str(data.get("final_tani", model_prediction)).upper().strip()
        yorum = str(data.get("klinik_yorum", "")).strip()

        if final_tani not in ("NORMAL", "ABNORMAL"):
            final_tani = model_prediction

        if not yorum:
            yorum = "Yorum alınamadı."

        # ✅ Sadece yanlışsa düzelt
        final_sonuc = model_prediction if model_ok else final_tani

        return {
            "final_sonuc": final_sonuc,
            "duzeltme_yapildi": (not model_ok),
            "yorum": yorum
        }

    except Exception as e:
        print(f"Gemini Hatası: {e}")
        return {
            "final_sonuc": model_prediction,
            "duzeltme_yapildi": False,
            "yorum": "AI doğrulaması hata verdi, model sonucu gösteriliyor."
        }

def find_leads_ocr(img):
    h, w = img.shape[:2]
    results = reader.readtext(img)
    leads = {}
    for (bbox, text, conf) in results:
        text = text.strip().upper().replace(" ", "")
        for lead in LEAD_NAMES:
            if lead.upper() == text or lead.upper() in text:
                x2 = int(max(p[0] for p in bbox))
                y1 = int(min(p[1] for p in bbox))
                y2 = int(max(p[1] for p in bbox))
                signal_x1 = x2 + 5
                signal_x2 = min(w, x2 + 300)
                signal_y1 = max(0, y1 - 20)
                signal_y2 = min(h, y2 + 20)
                if signal_x2 > signal_x1:
                    leads[lead] = img[signal_y1:signal_y2, signal_x1:signal_x2]
                break
    return leads

def predict_lead(lead_img):
    rgb = cv2.cvtColor(lead_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    return probs[0].item()

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yok'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 1. EKG Kontrolü
    is_ecg, _ = check_if_ecg(filepath)
    if not is_ecg:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Hata: Dosya silinemedi: {e}")
        return jsonify({'error': 'Bu bir EKG değil.'})

    # 2. Görüntü İşleme ve Model Tahmini
    img = cv2.imread(filepath)
    if img is None:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Hata: Dosya silinemedi: {e}")
        return jsonify({'error': 'Görsel okunamadı'})

    leads = find_leads_ocr(img)

    if not leads:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Hata: Dosya silinemedi: {e}")
        return jsonify({'error': 'Lead bulunamadı.'})

    probs = []
    for _, lead_img in leads.items():
        probs.append(predict_lead(lead_img))

    avg_conf = np.mean(probs)
    model_score = round(avg_conf * 100, 1)

    # Modelin ham tahmini
    model_prediction = "NORMAL" if avg_conf >= 0.55 else "ABNORMAL"

    # Modeli, skoru ve resmi Gemini'ye atıyoruz.
    gemini_result = verify_with_gemini(filepath, model_prediction, model_score)

    final_decision = gemini_result["final_sonuc"]
    ai_comment = gemini_result["yorum"]
    was_corrected = gemini_result["duzeltme_yapildi"]

    # ✅ Analiz sonrası resmi sil
    try:
        os.remove(filepath)
    except Exception as e:
        print(f"Hata: Dosya silinemedi: {e}")

    return jsonify({
        'genel_sonuc': final_decision,
        'model_ham_sonuc': model_prediction,
        'confidence': model_score,
        'ai_yorum': ai_comment,
        'duzeltildi_mi': was_corrected,
        'gemini_enabled': GEMINI_ENABLED
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)