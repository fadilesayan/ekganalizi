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

# Config
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

config = load_config()

# Gemini API
GEMINI_ENABLED = False
gemini_model = None

try:
    import google.generativeai as genai
    GEMINI_API_KEY = config.get('GEMINI_API_KEY', '')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        GEMINI_ENABLED = True
        print("✅ Gemini API aktif")
    else:
        print("⚠️ API key yok")
except Exception as e:
    print(f"⚠️ Gemini yüklenemedi: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Model yükle
print("Model yükleniyor...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load("ekg_models_resnet50/resnet50_ptbxl_gray_fold3.pth", map_location="cpu", weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print("✅ Model yüklendi")

# OCR yükle
print("OCR yükleniyor...")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR yüklendi")

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def check_if_ecg(img_path):
    """Gemini ile EKG kontrolü"""
    if not GEMINI_ENABLED:
        return True, ""
    
    try:
        img = Image.open(img_path)
        
        prompt = """Bu görüntü bir EKG (elektrokardiyogram) mı?

EKG özellikleri:
- Grid çizgili kağıt
- Kalp ritmi dalgaları
- Lead etiketleri (I, II, III, aVR, aVL, aVF, V1-V6)

SADECE şu formatta yanıt ver:
EKG: EVET veya HAYIR
AÇIKLAMA: (kısa)
"""
        
        response = gemini_model.generate_content([prompt, img])
        text = response.text.upper()
        
        if "EKG: HAYIR" in text or "EKG:HAYIR" in text:
            return False, response.text
        return True, response.text
        
    except Exception as e:
        return True, str(e)

def get_gemini_analysis(img_path, result, score):
    """Gemini ile DENGELI EKG analizi"""
    if not GEMINI_ENABLED:
        return "Gemini API aktif değil"
    
    try:
        img = Image.open(img_path)
        
        # Sonuca göre farklı prompt
        if result == "NORMAL":
            prompt = f"""Bu EKG'yi değerlendir. Model sonucu: NORMAL (%{score} güven)

2-3 CÜMLE ile OLUMLU ve RAHATLATICI yorum yap:
- Ritim düzenli mi?
- Kalp hızı normal aralıkta mı?
- Genel durum iyi mi?

TÜRKÇE, OLUMLU TON, KISA!
Örnek: "EKG normal sinüs ritmi gösteriyor. Kalp hızı düzenli ve normal aralıkta. Genel görünüm iyi ancak rutin kontroller ihmal edilmemelidir."
"""
        else:
            prompt = f"""Bu EKG'yi değerlendir. Model sonucu: ABNORMAL (%{score} güven)

2-3 CÜMLE ile SAKIN ve YÖNLENDİRİCİ yorum yap:
- Ne gibi bulgular var?
- Acil mi, yoksa kontrol edilmeli mi?
- Nasıl bir öneri verilmeli?

TÜRKÇE, PANİK YARATMADAN, KISA!
Örnek: "EKG'de bazı düzensizlikler tespit edildi. Bu bulgular mutlaka bir kardiyolog tarafından değerlendirilmelidir. En kısa sürede doktora başvurmanız önerilir."
"""
        
        response = gemini_model.generate_content([prompt, img])
        return response.text
        
    except Exception as e:
        return f"AI analizi yapılamadı: {str(e)}"

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
    
    # 1. EKG mi kontrol et
    is_ecg, ecg_msg = check_if_ecg(filepath)
    if not is_ecg:
        return jsonify({
            'error': 'Bu bir EKG görüntüsü değil! Lütfen gerçek EKG yükleyin.',
            'details': ecg_msg
        })
    
    # 2. Görüntüyü oku
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Görsel okunamadı'})
    
    # 3. OCR ile lead bul
    leads = find_leads_ocr(img)
    
    if len(leads) == 0:
        return jsonify({'error': 'Lead bulunamadı. Görüntü kalitesini kontrol edin.'})
    
    # 4. Model ile analiz
    probs = []
    for name, lead_img in leads.items():
        p = predict_lead(lead_img)
        probs.append(p)
    
    avg = np.mean(probs)
    score = round(avg * 100, 1)
    
    THR = 0.55
    result = "NORMAL" if avg >= THR else "ABNORMAL"
    
    # 5. Gemini analizi
    ai_analysis = get_gemini_analysis(filepath, result, score)

    # ✅ SADECE EKLENEN: dosyayı sil
    os.remove(filepath)
    
    # Frontend için format (orijinal tasarım)
    return jsonify({
        'genel_sonuc': result,
        'confidence': score,
        'normal_score': score if result == "NORMAL" else 100 - score,
        'abnormal_score': 100 - score if result == "NORMAL" else score,
        'ai_yorum': ai_analysis,
        'gemini_enabled': GEMINI_ENABLED
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
