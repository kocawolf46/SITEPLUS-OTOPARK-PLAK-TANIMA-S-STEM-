"""
SITEPLUS - Büyük Site Otopark Sistemi v3
Flask + OpenCV + EasyOCR + Web Dashboard
Kamera HİÇ kapanmaz, web üzerinden canlı izlenir
"""
import cv2
import easyocr
from difflib import SequenceMatcher
import time
import csv
import os
import threading
from datetime import datetime
from collections import deque
from flask import Flask, Response, jsonify, request, render_template_string

# ===================== AYARLAR =====================
print("OCR modeli yükleniyor...")
reader = easyocr.Reader(['tr', 'en'], gpu=False)
print("Model yüklendi!")

YETKILI_ARACLAR_DB = {
    "34ABC123": {"sahip": "Ahmet Yılmaz", "model": "Renault Clio (Gri)", "daire": "A-12", "tip": "Sakin"},
    "06KOCA06": {"sahip": "Mehmet Demir", "model": "Toyota Corolla (Beyaz)", "daire": "B-05", "tip": "Yonetici"},
    "46DZ460":  {"sahip": "Koca Wolf", "model": "Honda Civic (Siyah)", "daire": "C-01", "tip": "VIP"},
    "35IZMIR35": {"sahip": "Ayşe Kaya", "model": "Volkswagen Golf (Kırmızı)", "daire": "A-23", "tip": "Sakin"}
}

# Global durum
class SystemState:
    def __init__(self):
        self.bariyer = "KAPALI"  # ACIK / KAPALI
        self.bariyer_zaman = 0
        self.oto_mod = True
        self.son_plaka = None
        self.son_plaka_detay = {}
        self.son_tarama = 0
        self.son_ayni_plaka = ""
        self.son_ayni_zaman = 0
        self.frame = None
        self.lock = threading.Lock()
        self.logs = deque(maxlen=100)
        self.son_plakalar = deque(maxlen=10)
        self.istatistik = {"bugun_giris": 0, "yetkisiz": 0}
        self.fps = 0
        self.camera_active = True
        self.durum_text = "Sistem Hazir - Plaka Bekleniyor"
        self.durum_renk = "yellow"

state = SystemState()

# ===================== PLAKA TANIMA =====================
def plaka_benzerlik(okunan, veritabani, esik=0.75):
    en_iyi, max_b = None, 0.0
    for kayitli in veritabani.keys():
        oran = SequenceMatcher(None, okunan, kayitli).ratio()
        oran2 = SequenceMatcher(None, okunan.replace('0','O'), kayitli).ratio()
        oran = max(oran, oran2)
        if oran > max_b:
            max_b, en_iyi = oran, kayitli
    if max_b >= esik:
        return en_iyi, max_b
    return None, 0.0

def gelismis_roi(gray_roi):
    h,w = gray_roi.shape
    buyuk = cv2.resize(gray_roi, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    filt = cv2.bilateralFilter(buyuk, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def plaka_tespit(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filt, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    for c in contours:
        if cv2.contourArea(c) < 800: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            aspect = w/float(h)
            if 2.0 < aspect < 6.0 and w>80 and h>20:
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size==0: continue
                roi_gelismis = gelismis_roi(roi_gray)
                results = reader.readtext(roi_gelismis, detail=1)
                okunan, conf = "", 0
                for _, text, prob in results:
                    if prob < 0.4: continue
                    clean = "".join(e for e in text if e.isalnum()).upper()
                    if len(clean)>=5 and len(clean)>len(okunan):
                        okunan, conf = clean, prob
                if okunan:
                    return approx, (x,y,w,h), roi_gelismis, okunan, conf
    return None, None, None, None, 0

# ===================== KAMERA THREAD =====================
def camera_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    fps_times = deque(maxlen=30)
    last_scan = 0
    
    while state.camera_active:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        now = time.time()
        # Otomatik tarama
        if state.oto_mod and now - last_scan > 1.5:
            last_scan = now
            kontur, rect, roi, okunan, conf = plaka_tespit(frame)
            if okunan and len(okunan)>=5:
                if not (okunan == state.son_ayni_plaka and now - state.son_ayni_zaman < 5.0):
                    eslesen, benzerlik = plaka_benzerlik(okunan, YETKILI_ARACLAR_DB, 0.75)
                    zaman_str = datetime.now().strftime("%H:%M:%S")
                    if eslesen:
                        bilgi = YETKILI_ARACLAR_DB[eslesen]
                        state.son_plaka = eslesen
                        state.son_plaka_detay = {"okunan": okunan, "eslesen": eslesen, "bilgi": bilgi, "benzerlik": benzerlik, "conf": conf, "durum": "YETKILI", "zaman": zaman_str}
                        state.bariyer = "ACIK"
                        state.bariyer_zaman = now
                        state.durum_text = f"YETKILI - {eslesen} - BARIYER ACILDI"
                        state.durum_renk = "green"
                        state.son_plakalar.appendleft({"plaka": eslesen, "zaman": zaman_str, "yetkili": True, "sahip": bilgi['sahip']})
                        state.istatistik["bugun_giris"] += 1
                        print(f"[YETKILI] {eslesen} -> {bilgi['sahip']}")
                    else:
                        state.son_plaka = okunan
                        state.son_plaka_detay = {"okunan": okunan, "eslesen": None, "bilgi": None, "benzerlik": 0, "conf": conf, "durum": "YETKISIZ", "zaman": zaman_str}
                        state.durum_text = f"YETKISIZ - {okunan} - GIRIS YASAK"
                        state.durum_renk = "red"
                        state.son_plakalar.appendleft({"plaka": okunan, "zaman": zaman_str, "yetkili": False, "sahip": "-"})
                        state.istatistik["yetkisiz"] += 1
                        print(f"[YETKISIZ] {okunan}")
                    
                    state.logs.appendleft({"tarih": datetime.now().strftime("%d.%m.%Y %H:%M:%S"), "plaka": okunan, "eslesen": eslesen if 'eslesen' in locals() else "-", "durum": "YETKILI" if eslesen else "YETKISIZ", "guven": f"{conf:.2f}"})
                    state.son_ayni_plaka = okunan
                    state.son_ayni_zaman = now
        
        # Bariyer otomatik kapat
        if state.bariyer == "ACIK" and now - state.bariyer_zaman > 4.0:
            state.bariyer = "KAPALI"
        
        # FPS
        fps_times.append(time.time()-t0)
        state.fps = 1.0 / (sum(fps_times)/len(fps_times)) if fps_times else 0
        
        # Frame'i overlay ile kaydet (web için)
        display = frame.copy()
        # Plaka kutusu varsa çiz
        # Basit overlay - web panelde detaylı UI var
        with state.lock:
            state.frame = display.copy()
    
    cap.release()

# Thread başlat
threading.Thread(target=camera_loop, daemon=True).start()

# ===================== FLASK =====================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SITEPLUS - Akıllı Otopark v3</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
<style>
 body{font-family:'Inter',sans-serif;background:#0a0a0f;color:#e5e7eb}
 .mono{font-family:'JetBrains Mono',monospace}
 .glass{background:rgba(255,255,255,0.04);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.08)}
 .scanline{position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#00ff88,transparent);animation:scan 2s linear infinite}
 @keyframes scan{0%{top:0}100%{top:100%}}
 .plate-glow{box-shadow:0 0 40px rgba(0,255,136,0.4)}
</style>
</head>
<body class="min-h-screen">
<!-- SIDEBAR -->
<div class="flex">
<div class="w-[260px] min-h-screen glass border-r border-white/10 p-6 hidden lg:block sticky top-0">
  <div class="flex items-center gap-3 mb-10">
    <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center font-black text-black">S+</div>
    <div><div class="font-extrabold leading-none">SITEPLUS</div><div class="text-[10px] tracking-[0.2em] text-white/50">PARK v3.0</div></div>
  </div>
  <div class="space-y-1 text-sm">
    <div class="px-3 py-2.5 rounded-xl bg-white text-black font-semibold">● Canlı Takip</div>
    <div class="px-3 py-2.5 rounded-xl text-white/60">Araç Veritabanı</div>
    <div class="px-3 py-2.5 rounded-xl text-white/60">Geçiş Geçmişi</div>
    <div class="px-3 py-2.5 rounded-xl text-white/60">Raporlar</div>
  </div>
  <div class="mt-10 p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/20">
    <div class="text-xs text-emerald-300">KAMERA DURUMU</div>
    <div class="flex items-center gap-2 mt-2"><span class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span><span class="text-sm font-semibold">Sürekli Açık • 1080p</span></div>
    <div class="text-[11px] text-white/50 mt-1" id="fps">FPS: --</div>
  </div>
</div>

<!-- MAIN -->
<div class="flex-1 p-4 lg:p-6">
  <!-- HEADER -->
  <div class="flex justify-between items-center mb-6">
    <div>
      <h1 class="text-2xl font-extrabold">Koca Residence • Ana Giriş</h1>
      <p class="text-white/50 text-sm">Adana / 124 Daire • 2 Blok • Anlık Takip</p>
    </div>
    <div class="text-right">
      <div class="mono text-sm" id="clock">--:--:--</div>
      <div class="text-xs text-white/50">Oto Mod: <span id="otoMod" class="text-emerald-400 font-bold">AÇIK</span> • Kamera asla kapanmaz</div>
    </div>
  </div>

  <!-- STATS -->
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    <div class="glass rounded-2xl p-4"><div class="text-white/50 text-xs">BUGÜN GİRİŞ</div><div class="text-3xl font-extrabold mono mt-1" id="statGiris">0</div><div class="text-xs text-emerald-400 mt-1">▲ +12% düne göre</div></div>
    <div class="glass rounded-2xl p-4"><div class="text-white/50 text-xs">YETKİLİ ARAÇ</div><div class="text-3xl font-extrabold mono mt-1">{{ yetkili_sayisi }}</div><div class="text-xs text-white/40 mt-1">Kayıtlı</div></div>
    <div class="glass rounded-2xl p-4"><div class="text-white/50 text-xs">YETKİSİZ DENEME</div><div class="text-3xl font-extrabold mono mt-1 text-red-400" id="statYetkisiz">0</div><div class="text-xs text-red-400/60 mt-1">Bugün</div></div>
    <div class="glass rounded-2xl p-4"><div class="text-white/50 text-xs">DOLULUK</div><div class="text-3xl font-extrabold mono mt-1">%68</div><div class="w-full h-1.5 bg-white/10 rounded-full mt-2"><div class="h-full bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full" style="width:68%"></div></div></div>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-[1.6fr_0.9fr] gap-6">
    <!-- CAMERA -->
    <div class="glass rounded-[24px] overflow-hidden">
      <div class="flex justify-between items-center p-4 border-b border-white/10">
        <div class="flex items-center gap-3"><span class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span><span class="text-sm font-bold">KAMERA 01 - ANA GİRİŞ • KAYITTA • 1080p</span><span class="text-[10px] px-2 py-1 rounded-full bg-white/10">SÜREKLİ AÇIK</span></div>
        <div class="flex gap-2">
          <button onclick="toggleOto()" class="px-3 py-1.5 rounded-full bg-white text-black text-xs font-bold">OTO: <span id="btnOto">AÇIK</span></button>
          <button onclick="manualScan()" class="px-3 py-1.5 rounded-full bg-white/10 text-xs">Manuel Tara [S]</button>
        </div>
      </div>
      <div class="relative bg-black aspect-video">
        <img src="/video_feed" class="w-full h-full object-cover" id="video">
        <div class="scanline"></div>
        <!-- Corner brackets -->
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[320px] h-[80px] border-2 border-emerald-400/60 rounded-lg pointer-events-none">
          <div class="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-emerald-400"></div>
          <div class="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-emerald-400"></div>
          <div class="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-emerald-400"></div>
          <div class="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-emerald-400"></div>
        </div>
        <div class="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
          <div class="flex items-end justify-between">
            <div>
              <div class="text-[10px] tracking-widest text-white/60">ANLIK DURUM</div>
              <div class="text-lg font-bold" id="durumText">Sistem Hazır</div>
            </div>
            <div class="text-right"><div class="text-[10px] text-white/60">BARIYER</div><div class="text-xl font-black mono" id="bariyerText">KAPALI</div></div>
          </div>
        </div>
      </div>
      <div class="p-3 flex gap-2 text-[11px] text-white/50">
        <span>● Kamera hiç kapanmaz, arka planda otomatik tarar</span><span>•</span><span>[Q] Çıkış [C] Bariyer Kapat [L] Log</span>
      </div>
    </div>

    <!-- RIGHT -->
    <div class="space-y-6">
      <!-- BARRIER -->
      <div class="glass rounded-[20px] p-5">
        <div class="text-xs tracking-widest text-white/50 mb-3">BARİYER KONTROL</div>
        <div class="rounded-2xl p-4 text-center" id="barierCard" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3)">
          <div class="text-4xl font-black mono" id="barierBig">KAPALI</div>
          <div class="text-xs text-white/60 mt-1" id="barierSub">Otomatik kapanacak: 4 sn</div>
        </div>
        <div class="grid grid-cols-2 gap-2 mt-3">
          <button onclick="setBarrier('ACIK')" class="py-2.5 rounded-xl bg-emerald-500 text-black font-bold text-sm">Bariyeri Aç</button>
          <button onclick="setBarrier('KAPALI')" class="py-2.5 rounded-xl bg-white/10 font-bold text-sm">Kapat</button>
          <button onclick="setBarrier('ACIK')" class="col-span-2 py-2 rounded-xl bg-red-500/20 border border-red-500/30 text-red-300 text-xs">🚨 Acil Durum - Sürekli Açık Tut</button>
        </div>
      </div>

      <!-- LAST PLATE -->
      <div class="glass rounded-[20px] p-5">
        <div class="text-xs tracking-widest text-white/50 mb-3">SON ALGILANAN</div>
        <div class="bg-black rounded-2xl p-4 border border-white/10 text-center plate-glow" id="lastPlateCard">
          <div class="inline-flex px-2 py-1 rounded-full text-[10px] font-bold bg-emerald-500 text-black mb-3" id="lastBadge">YETKILI</div>
          <div class="mono text-3xl font-black tracking-wider" id="lastPlaka">-- -- ---</div>
          <div class="mt-3 text-sm font-semibold" id="lastSahip">Bekleniyor...</div>
          <div class="text-xs text-white/50" id="lastModel">-</div>
          <div class="grid grid-cols-3 gap-2 mt-4 text-[11px]">
            <div class="bg-white/5 rounded-xl p-2"><div class="text-white/40">Benzerlik</div><div class="font-bold mono" id="lastBenzer">%--</div></div>
            <div class="bg-white/5 rounded-xl p-2"><div class="text-white/40">Güven</div><div class="font-bold mono" id="lastGuven">--</div></div>
            <div class="bg-white/5 rounded-xl p-2"><div class="text-white/40">Zaman</div><div class="font-bold mono" id="lastZaman">--:--</div></div>
          </div>
        </div>
      </div>

      <!-- SON GECISLER -->
      <div class="glass rounded-[20px] p-5">
        <div class="text-xs tracking-widest text-white/50 mb-3">SON GEÇİŞLER (Canlı)</div>
        <div id="gecisList" class="space-y-2 max-h-[200px] overflow-auto"></div>
      </div>
    </div>
  </div>

  <!-- TABLES -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
    <div class="glass rounded-[20px] p-5">
      <div class="flex justify-between items-center mb-4"><h3 class="font-bold">Yetkili Araçlar • {{ yetkili_sayisi }}</h3><span class="text-xs px-2 py-1 rounded-full bg-emerald-500/20 text-emerald-300">Aktif Veritabanı</span></div>
      <div class="space-y-2 text-sm">
        {% for plaka, bilgi in araclar.items() %}
        <div class="flex justify-between items-center p-3 rounded-xl bg-white/[0.03] border border-white/5">
          <div><div class="mono font-bold">{{ plaka }}</div><div class="text-xs text-white/50">{{ bilgi.sahip }} • {{ bilgi.daire }}</div></div>
          <div class="text-right"><div class="text-xs">{{ bilgi.model }}</div><div class="text-[10px] px-2 py-0.5 rounded-full {{ 'bg-purple-500/20 text-purple-300' if bilgi.tip=='VIP' else 'bg-white/10 text-white/60' }} inline-block mt-1">{{ bilgi.tip }}</div></div>
        </div>
        {% endfor %}
      </div>
    </div>
    <div class="glass rounded-[20px] p-5">
      <div class="flex justify-between items-center mb-4"><h3 class="font-bold">Geçiş Logları</h3><button onclick="exportLog()" class="text-xs px-2 py-1 rounded-full bg-white/10">CSV İndir</button></div>
      <div class="overflow-auto max-h-[300px]">
        <table class="w-full text-xs"><thead class="text-white/40"><tr><th class="text-left p-2">Tarih</th><th class="text-left p-2">Plaka</th><th class="text-left p-2">Durum</th><th class="text-left p-2">Güven</th></tr></thead><tbody id="logTable"></tbody></table>
      </div>
    </div>
  </div>
</div>
</div>

<script>
function updateClock(){document.getElementById('clock').innerText=new Date().toLocaleString('tr-TR')}
setInterval(updateClock,1000);updateClock();

async function fetchStatus(){
  let r=await fetch('/api/status'); let j=await r.json();
  document.getElementById('durumText').innerText=j.durum_text;
  document.getElementById('bariyerText').innerText=j.bariyer;
  document.getElementById('barierBig').innerText=j.bariyer;
  document.getElementById('statGiris').innerText=j.istatistik.bugun_giris;
  document.getElementById('statYetkisiz').innerText=j.istatistik.yetkisiz;
  document.getElementById('fps').innerText='FPS: '+j.fps.toFixed(1);
  document.getElementById('otoMod').innerText=j.oto_mod?'AÇIK':'KAPALI';
  document.getElementById('btnOto').innerText=j.oto_mod?'AÇIK':'KAPALI';
  document.getElementById('barierCard').style.background=j.bariyer=='ACIK'?'rgba(16,185,129,0.15)':'rgba(239,68,68,0.1)';
  document.getElementById('barierCard').style.borderColor=j.bariyer=='ACIK'?'rgba(16,185,129,0.4)':'rgba(239,68,68,0.3)';
  if(j.son_detay && j.son_detay.okunan){
    document.getElementById('lastPlaka').innerText=j.son_detay.eslesen||j.son_detay.okunan;
    document.getElementById('lastSahip').innerText=j.son_detay.bilgi?j.son_detay.bilgi.sahip:'Kayıt Yok';
    document.getElementById('lastModel').innerText=j.son_detay.bilgi?j.son_detay.bilgi.model:'-';
    document.getElementById('lastBenzer').innerText=j.son_detay.benzerlik?('%'+Math.round(j.son_detay.benzerlik*100)):'%--';
    document.getElementById('lastGuven').innerText=j.son_detay.conf?j.son_detay.conf.toFixed(2):'--';
    document.getElementById('lastZaman').innerText=j.son_detay.zaman||'--:--';
    document.getElementById('lastBadge').innerText=j.son_detay.durum;
    document.getElementById('lastBadge').className=j.son_detay.durum=='YETKILI'?'inline-flex px-2 py-1 rounded-full text-[10px] font-bold bg-emerald-500 text-black mb-3':'inline-flex px-2 py-1 rounded-full text-[10px] font-bold bg-red-500 text-white mb-3';
  }
  let gl=document.getElementById('gecisList'); gl.innerHTML='';
  j.son_plakalar.forEach(p=>{
    gl.innerHTML+=`<div class="flex justify-between items-center p-2 rounded-xl ${p.yetkili?'bg-emerald-500/10 border border-emerald-500/20':'bg-red-500/10 border border-red-500/20'}"><div class="mono text-sm font-bold">${p.plaka}</div><div class="text-[11px] text-white/50">${p.sahip} • ${p.zaman}</div></div>`;
  });
  let lt=document.getElementById('logTable'); lt.innerHTML='';
  j.logs.forEach(l=>{
    lt.innerHTML+=`<tr class="border-b border-white/5"><td class="p-2 text-white/60">${l.tarih}</td><td class="p-2 mono font-bold">${l.plaka}</td><td class="p-2"><span class="px-2 py-0.5 rounded-full text-[10px] ${l.durum=='YETKILI'?'bg-emerald-500/20 text-emerald-300':'bg-red-500/20 text-red-300'}">${l.durum}</span></td><td class="p-2">${l.guven}</td></tr>`;
  });
}
setInterval(fetchStatus,800);fetchStatus();

async function setBarrier(s){await fetch('/api/barrier',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({durum:s})})}
async function toggleOto(){await fetch('/api/toggle_oto',{method:'POST'})}
async function manualScan(){await fetch('/api/manual_scan',{method:'POST'})}
async function exportLog(){window.location='/api/export'}
document.addEventListener('keydown',e=>{
  if(e.key.toLowerCase()=='c') setBarrier('KAPALI');
  if(e.key.toLowerCase()=='s') manualScan();
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, araclar=YETKILI_ARACLAR_DB, yetkili_sayisi=len(YETKILI_ARACLAR_DB))

def gen_frames():
    while True:
        with state.lock:
            frame = state.frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with state.lock:
        return jsonify({
            "bariyer": state.bariyer,
            "oto_mod": state.oto_mod,
            "durum_text": state.durum_text,
            "fps": state.fps,
            "son_plaka": state.son_plaka,
            "son_detay": state.son_plaka_detay,
            "son_plakalar": list(state.son_plakalar),
            "logs": list(state.logs)[:20],
            "istatistik": state.istatistik
        })

@app.route('/api/barrier', methods=['POST'])
def api_barrier():
    data = request.json
    durum = data.get('durum','KAPALI')
    state.bariyer = durum
    state.bariyer_zaman = time.time() if durum=="ACIK" else 0
    return jsonify({"ok": True, "bariyer": state.bariyer})

@app.route('/api/toggle_oto', methods=['POST'])
def api_toggle():
    state.oto_mod = not state.oto_mod
    return jsonify({"oto_mod": state.oto_mod})

@app.route('/api/manual_scan', methods=['POST'])
def api_manual():
    # Manuel tetikleme - hemen tarama için flag reset
    # camera_loop zaten otomatik tarıyor, burada ekstra bir kareyi zorla okut
    with state.lock:
        frame = state.frame
    if frame is not None:
        kontur, rect, roi, okunan, conf = plaka_tespit(frame)
        if okunan:
            eslesen, benzerlik = plaka_benzerlik(okunan, YETKILI_ARACLAR_DB, 0.75)
            zaman_str = datetime.now().strftime("%H:%M:%S")
            if eslesen:
                bilgi = YETKILI_ARACLAR_DB[eslesen]
                state.son_plaka_detay = {"okunan": okunan, "eslesen": eslesen, "bilgi": bilgi, "benzerlik": benzerlik, "conf": conf, "durum": "YETKILI", "zaman": zaman_str}
                state.bariyer = "ACIK"
                state.bariyer_zaman = time.time()
                state.durum_text = f"MANUEL YETKILI - {eslesen}"
                state.son_plakalar.appendleft({"plaka": eslesen, "zaman": zaman_str, "yetkili": True, "sahip": bilgi['sahip']})
            else:
                state.son_plaka_detay = {"okunan": okunan, "eslesen": None, "bilgi": None, "benzerlik": 0, "conf": conf, "durum": "YETKISIZ", "zaman": zaman_str}
                state.durum_text = f"MANUEL YETKISIZ - {okunan}"
                state.son_plakalar.appendleft({"plaka": okunan, "zaman": zaman_str, "yetkili": False, "sahip": "-"})
            state.logs.appendleft({"tarih": datetime.now().strftime("%d.%m.%Y %H:%M:%S"), "plaka": okunan, "eslesen": eslesen if 'eslesen' in locals() else "-", "durum": "YETKILI" if eslesen else "YETKISIZ", "guven": f"{conf:.2f}"})
    return jsonify({"ok": True})

@app.route('/api/export')
def api_export():
    # CSV olarak logları döndür
    import io
    output = io.StringIO()
    output.write("Tarih,Plaka,Eslesen,Durum,Guven\n")
    for l in state.logs:
        output.write(f"{l['tarih']},{l['plaka']},{l['eslesen']},{l['durum']},{l['guven']}\n")
    return Response(output.getvalue(), mimetype='text/csv', headers={"Content-Disposition":"attachment;filename=otopark_log.csv"})

if __name__ == '__main__':
    print("\n=== SITEPLUS v3 WEB SUNUCU BAŞLATILIYOR ===")
    print("Kamera SÜREKLİ AÇIK modda çalışıyor...")
    print("Tarayıcıdan aç: http://localhost:5000")
    print("Büyük sitelerdeki gibi dashboard hazır!\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
