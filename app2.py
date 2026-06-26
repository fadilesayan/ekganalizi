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

# ======================
# Config
# ======================
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

config = load_config()

# ======================
# Gemini API
# ======================
GEMINI_ENABLED = False
gemini_model = None

try:
    import google.generativeai as genai
    GEMINI_API_KEY = config.get('GEMINI_API_KEY', '')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        GEMINI_ENABLED = True
        print("✅ Gemini API aktif")
    else:
        print("⚠️ API key yok")
except Exception as e:
    print(f"⚠️ Gemini yüklenemedi: {e}")

# ======================
# Flask
# ======================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# ======================
# Model yükle
# ======================
print("Model yükleniyor...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

state = torch.load(
    "ekg_models_resnet50/resnet50_ptbxl_gray_fold3.pth",
    map_location="cpu",
    weights_only=False
)

if "state_dict" in state:
    state = state["state_dict"]

state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print("✅ Model yüklendi")

# ======================
# OCR
# ======================
print("OCR yükleniyor...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR yüklendi")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ======================
# Helpers
# ======================
def check_if_ecg(img_path):
    if not GEMINI_ENABLED:
        return True, ""

    try:
        img = Image.open(img_path)
        prompt = """Bu görüntü bir EKG mi?

SADECE şu formatta yanıt ver:
EKG: EVET veya HAYIR
"""
        response = gemini_model.generate_content([prompt, img])
        text = response.text.upper()

        if "EKG: HAYIR" in text:
            return False, response.text
        return True, response.text

    except Exception as e:
        return True, str(e)


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

                sx1 = x2 + 5
                sx2 = min(w, x2 + 300)
                sy1 = max(0, y1 - 20)
                sy2 = min(h, y2 + 20)

                if sx2 > sx1:
                    leads[lead] = img[sy1:sy2, sx1:sx2]
                break

    return leads


def predict_lead(lead_img):
    rgb = cv2.cvtColor(lead_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]

    return probs[0].item()


# ======================
# Gemini uzun yorum (5–6 cümle)
# ======================
def analyze_with_gemini(img_path, model_result, model_score, lead_details):
    if not GEMINI_ENABLED:
        return model_result, "Gemini API aktif değil", False, ""

    try:
        img = Image.open(img_path)

        prompt = f"""Bu bir 12-lead EKG görüntüsü.

Model sonucu: {model_result}
Model güveni: %{model_score}

KURALLAR:
- KARAR satırından sonra TEK PARAGRAF yaz.
- 5 ila 6 cümle uzunluğunda olsun.
- ÖNERİ, NOT, RİTİM, ST, BULGU gibi başlıklar KULLANMA.
- Madde işareti veya listeleme YAPMA.
- Akıcı, sade ve tıbbi ama anlaşılır Türkçe kullan.
- Panik yaratma, kesin teşhis koyduğunu iddia etme.

FORMAT (birebir):
KARAR: NORMAL veya ABNORMAL
YORUM: 5-6 cümlelik tek paragraf açıklama
"""

        response = gemini_model.generate_content([prompt, img])
        ai_text = response.text.strip()

        if "KARAR:" in ai_text.upper():
            karar_line = ai_text.upper().split("KARAR:")[-1].split("\n")[0]
            ai_result = "ABNORMAL" if "ABNORMAL" in karar_line else "NORMAL"
        else:
            ai_result = model_result

        corrected = (ai_result != model_result)

        return ai_result, ai_text, corrected, ai_text

    except Exception as e:
        return model_result, f"Gemini hatası: {str(e)}", False, ""


# ======================
# Routes
# ======================
@app.route('/')
def index():
    return render_template('index2.html', gemini_enabled=GEMINI_ENABLED)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yok'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    is_ecg, _ = check_if_ecg(filepath)
    if not is_ecg:
        return jsonify({'error': 'Bu görüntü EKG değil'})

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Görsel okunamadı'})

    leads = find_leads_ocr(img)
    if len(leads) == 0:
        return jsonify({'error': 'Lead bulunamadı'})

    probs = []
    for _, lead_img in leads.items():
        probs.append(predict_lead(lead_img))

    avg = np.mean(probs)
    model_score = round(avg * 100, 1)

    THR = 0.52
    model_result = "NORMAL" if avg >= THR else "ABNORMAL"

    final_result, ai_explanation, _, _ = analyze_with_gemini(
        filepath, model_result, model_score, None
    )

    return jsonify({
        'final_result': final_result,
        'confidence': model_score,
        'ai_yorum': ai_explanation,
        'gemini_enabled': GEMINI_ENABLED
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
