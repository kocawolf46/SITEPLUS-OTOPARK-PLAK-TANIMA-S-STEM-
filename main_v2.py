import cv2
import easyocr
from difflib import SequenceMatcher
import time
import csv
import os
from datetime import datetime
from collections import deque

print("OCR modeli yükleniyor...")
reader = easyocr.Reader(['tr', 'en'], gpu=False)
print("Model başarıyla yüklendi!")

# ===================== AYARLAR =====================
YETKILI_ARACLAR_DB = {
    "34ABC123": {"sahip": "Ahmet Yılmaz", "model": "Renault Clio (Gri)", "tip": "Sakin"},
    "06KOCA06": {"sahip": "Mehmet Demir", "model": "Toyota Corolla (Beyaz)", "tip": "Yonetici"},
    "46DZ460":  {"sahip": "Koca Wolf", "model": "Honda Civic (Siyah)", "tip": "VIP"},
    "35IZMIR35": {"sahip": "Ayşe Kaya", "model": "Volkswagen Golf (Kırmızı)", "tip": "Sakin"}
}

# Sistem parametreleri
AUTO_SCAN_INTERVAL = 1.5  # saniye - otomatik modda tarama aralığı
COOLDOWN_SURESI = 5.0      # aynı plaka için tekrar okumama süresi
BENZERLIK_ESIGI = 0.75
LOG_DOSYASI = "otopark_log.csv"
CONFIDENCE_ESIGI = 0.4

# ===================== YARDIMCI FONKSIYONLAR =====================

def log_kaydi_ekle(plaka, durum, sahip, benzerlik, confidence):
    dosya_var = os.path.exists(LOG_DOSYASI)
    with open(LOG_DOSYASI, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not dosya_var:
            writer.writerow(["Tarih", "Saat", "Okunan Plaka", "Durum", "Sahip", "Benzerlik %", "OCR Guveni"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            plaka, durum, sahip, int(benzerlik*100), round(confidence, 2)
        ])

def plaka_benzerlik_kontrolu(okunan, veritabani, esik_orani=0.75):
    en_iyi_eslesme = None
    max_benzerlik = 0.0
    for kayitli_plaka in veritabani.keys():
        oran = SequenceMatcher(None, okunan, kayitli_plaka).ratio()
        # 0 -> O , 1 -> I toleransı için ek düzeltme
        duzeltilmis_okunan = okunan.replace('0','O').replace('1','I')
        oran2 = SequenceMatcher(None, duzeltilmis_okunan, kayitli_plaka).ratio()
        oran = max(oran, oran2)
        if oran > max_benzerlik:
            max_benzerlik = oran
            en_iyi_eslesme = kayitli_plaka
    if max_benzerlik >= esik_orani:
        return en_iyi_eslesme, max_benzerlik
    return None, 0.0

def gelismis_roi_on_isleme(gray_roi):
    """OCR için ROI'yi daha okunur hale getir"""
    # 1. Boyut büyüt
    h, w = gray_roi.shape
    buyutulmus = cv2.resize(gray_roi, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    # 2. Gürültü azalt
    filtered = cv2.bilateralFilter(buyutulmus, 11, 17, 17)
    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # 4. Biraz blur ile temizle
    return thresh

def plaka_tespit_ve_ocr(frame):
    """Frame içinden plaka bul ve OCR yap - gelişmiş versiyon"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    adaylar = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 800:  # çok küçük konturları ele
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            # Türk plakaları yaklaşık 3:1 - 5:1 oranında
            if 2.0 < aspect < 6.0 and w > 80 and h > 20:
                adaylar.append((approx, x, y, w, h))

    if not adaylar:
        return None, None, None, None, 0.0

    # En mantıklı adayı al
    plaka_konturu, x, y, w, h = adaylar[0]
    
    # ROI al ve geliştir
    plaka_roi_gray = gray[y:y+h, x:x+w]
    if plaka_roi_gray.size == 0:
        return None, None, None, None, 0.0
        
    gelismis_roi = gelismis_roi_on_isleme(plaka_roi_gray)

    # EasyOCR - detaylı sonuçlar
    results = reader.readtext(gelismis_roi, detail=1, paragraph=False)
    
    okunan_plaka = ""
    max_conf = 0.0
    for (bbox, text, prob) in results:
        if prob < CONFIDENCE_ESIGI:
            continue
        clean_text = "".join(e for e in text if e.isalnum()).upper()
        clean_text = clean_text.replace('0','O').replace('O','O')  # OCR hatalarını koru, sonra benzerlikte tolere et
        # Ham temizleme
        clean_text = "".join(c for c in clean_text if c.isalnum())
        if len(clean_text) >= 5:
            # En uzun ve güvenli olanı seç
            if len(clean_text) > len(okunan_plaka):
                okunan_plaka = clean_text
                max_conf = prob

    # İkinci deneme: orijinal ROI üzerinden
    if not okunan_plaka:
        results2 = reader.readtext(plaka_roi_gray, detail=1)
        for (bbox, text, prob) in results2:
            clean_text = "".join(e for e in text if e.isalnum()).upper()
            if len(clean_text) >= 5:
                okunan_plaka = clean_text
                max_conf = prob
                break

    return plaka_konturu, (x,y,w,h), gelismis_roi, okunan_plaka, max_conf

def overlay_ciz(frame, durum_text, detay_text, renk, bariyer_durum, fps, mod, son_plakalar):
    """Gelişmiş arayüz çizimi - kamera görüntüsünün üstüne"""
    h, w = frame.shape[:2]
    # Üst panel - yarı şeffaf
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 110), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"MOD: {mod} | FPS: {fps:.1f} | Bariyer: {bariyer_durum}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, durum_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
    cv2.putText(frame, detay_text, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Alt panel - kontroller
    cv2.rectangle(overlay, (0, h-60), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    kontroller = " [S] Manuel Tara | [A] Oto Mod Ac/Kapat | [L] Loglari Goster | [Q] Cikis | [C] Bariyeri Kapat"
    cv2.putText(frame, kontroller, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Sağ taraf - son plakalar listesi
    if son_plakalar:
        y0 = 130
        cv2.rectangle(frame, (w-280, y0-20), (w-10, y0+len(son_plakalar)*22+10), (30,30,30), -1)
        cv2.rectangle(frame, (w-280, y0-20), (w-10, y0+len(son_plakalar)*22+10), (100,100,100), 1)
        cv2.putText(frame, "SON GECISLER:", (w-270, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        for i, (plaka, zaman, ok) in enumerate(list(son_plakalar)[-5:]):
            renk2 = (0,255,0) if ok else (0,0,255)
            cv2.putText(frame, f"{zaman} {plaka}", (w-270, y0+18+(i*18)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, renk2, 1)

    return frame

def canli_kamera_akisi_gelismis():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Kamera açılamadı! 1 numaralı kamerayı deniyorum...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Hiç kamera bulunamadı!")
            return

    print("\n=== GELİŞMİŞ OTOPARK SİSTEMİ v2.0 AKTİF ===")
    print("Özellikler:")
    print("- Kamera HİÇ kapanmıyor, sürekli açık")
    print("- Otomatik modda kendi tarıyor")
    print("- Sonuçlar 3 saniye ekranda kalıp kayboluyor")
    print("- Yetkili araçta bariyer otomatik açılıyor")
    print("------------------------------------------------")

    oto_mod = True
    son_tarama_zamani = 0
    son_ayni_plaka = ""
    son_ayni_plaka_zamani = 0
    
    bariyer_durum = "KAPALI"
    bariyer_acilma_zamani = 0
    BARRIER_OTOMATIK_KAPANMA = 4.0

    aktif_durum_text = "Sistem Hazir - Plaka Bekleniyor"
    aktif_detay_text = f"Yetkili arac sayisi: {len(YETKILI_ARACLAR_DB)}"
    aktif_renk = (255, 255, 0)
    aktif_sonuc_zamani = 0
    SONUC_GOSTERME_SURESI = 3.5

    son_plakalar = deque(maxlen=10)
    fps_counter = deque(maxlen=30)

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame alınamadı!")
            break

        simdi = time.time()

        # Bariyer otomatik kapanma kontrolü
        if bariyer_durum == "ACIK" and simdi - bariyer_acilma_zamani > BARRIER_OTOMATIK_KAPANMA:
            bariyer_durum = "KAPALI"
            print(">>> Bariyer otomatik kapandı")

        # Otomatik tarama
        tarama_yap = False
        if oto_mod and simdi - son_tarama_zamani > AUTO_SCAN_INTERVAL:
            tarama_yap = True
            son_tarama_zamani = simdi

        # Manuel tarama tuşu için flag daha sonra key ile tetiklenecek, burada otomatik olanı yap
        if tarama_yap:
            kontur, rect, roi, okunan, conf = plaka_tespit_ve_ocr(frame)
            
            if okunan and len(okunan) >= 5:
                # Cooldown - aynı plakayı sürekli okuma
                if okunan == son_ayni_plaka and simdi - son_ayni_plaka_zamani < COOLDOWN_SURESI:
                    pass  # Aynı plakayı kısa süre içinde tekrar işleme
                else:
                    eslesen, benzerlik = plaka_benzerlik_kontrolu(okunan, YETKILI_ARACLAR_DB, BENZERLIK_ESIGI)
                    
                    if eslesen:
                        bilgi = YETKILI_ARACLAR_DB[eslesen]
                        aktif_durum_text = f"YETKILI ARAC - BARIYER ACILDI [{eslesen}]"
                        aktif_detay_text = f"{bilgi['sahip']} | {bilgi['model']} | Benzerlik %{int(benzerlik*100)}"
                        aktif_renk = (0, 255, 0)
                        bariyer_durum = "ACIK"
                        bariyer_acilma_zamani = simdi
                        son_plakalar.append((eslesen, datetime.now().strftime("%H:%M:%S"), True))
                        log_kaydi_ekle(okunan, "YETKILI", f"{bilgi['sahip']}", benzerlik, conf)
                        print(f"[YETKILI] {eslesen} -> {bilgi['sahip']} (Okunan: {okunan} | Benzerlik: %{int(benzerlik*100)} | Conf: {conf:.2f})")
                    else:
                        aktif_durum_text = f"YETKISIZ ARAC - GIRIS YASAK [{okunan}]"
                        aktif_detay_text = f"Kayit bulunamadi | Guven: {conf:.2f}"
                        aktif_renk = (0, 0, 255)
                        son_plakalar.append((okunan, datetime.now().strftime("%H:%M:%S"), False))
                        log_kaydi_ekle(okunan, "YETKISIZ", "-", 0, conf)
                        print(f"[YETKISIZ] {okunan} (Conf: {conf:.2f})")

                    son_ayni_plaka = okunan if 'eslesen' not in locals() or not eslesen else eslesen
                    son_ayni_plaka_zamani = simdi
                    aktif_sonuc_zamani = simdi

                    # Plaka konturunu geçici olarak çiz
                    if kontur is not None and rect is not None:
                        cv2.drawContours(frame, [kontur], -1, aktif_renk, 3)
                        x,y,w,h = rect
                        cv2.putText(frame, okunan, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # Sonuç gösterme süresi dolduysa resetle
        if simdi - aktif_sonuc_zamani > SONUC_GOSTERME_SURESI and aktif_sonuc_zamani != 0:
            aktif_durum_text = "Sistem Hazir - Plaka Bekleniyor"
            aktif_detay_text = f"Yetkili arac sayisi: {len(YETKILI_ARACLAR_DB)} | Oto Mod: {'ACIK' if oto_mod else 'KAPALI'}"
            aktif_renk = (255, 255, 0)
            aktif_sonuc_zamani = 0

        # FPS hesapla
        fps_counter.append(time.time() - t_start)
        fps = 1.0 / (sum(fps_counter)/len(fps_counter)) if fps_counter else 0

        # Arayüzü çiz - KAMERA HİÇ KAPANMIYOR
        mod_str = "OTO" if oto_mod else "MANUEL"
        frame = overlay_ciz(frame, aktif_durum_text, aktif_detay_text, aktif_renk, bariyer_durum, fps, mod_str, son_plakalar)

        # Bariyer görseli
        if bariyer_durum == "ACIK":
            cv2.rectangle(frame, (20, 120), (40, 250), (0,255,0), -1)
            cv2.putText(frame, "BARIYER ACIK", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.rectangle(frame, (20, 120), (200, 140), (0,0,255), -1)

        cv2.imshow("Gelismis Otopark Sistemi v2 - Kamera SUREKLI ACIK", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Sistem kapatiliyor...")
            break
        elif key == ord('s'):
            # Manuel tetikleme - aynı anda tarar
            print("\n[MANUEL] Kare yakalandi, taranıyor...")
            son_tarama_zamani = 0  # bir sonraki otoda hemen tarasın
            # Manuel taramayı zorla yap
            kontur, rect, roi, okunan, conf = plaka_tespit_ve_ocr(frame)
            if okunan:
                eslesen, benzerlik = plaka_benzerlik_kontrolu(okunan, YETKILI_ARACLAR_DB, BENZERLIK_ESIGI)
                if eslesen:
                    bilgi = YETKILI_ARACLAR_DB[eslesen]
                    aktif_durum_text = f"YETKILI (MANUEL) - {eslesen}"
                    aktif_detay_text = f"{bilgi['sahip']} | Benzerlik %{int(benzerlik*100)}"
                    aktif_renk = (0, 255, 0)
                    bariyer_durum = "ACIK"
                    bariyer_acilma_zamani = simdi
                else:
                    aktif_durum_text = f"YETKISIZ (MANUEL) - {okunan}"
                    aktif_detay_text = "Kayit yok"
                    aktif_renk = (0,0,255)
                aktif_sonuc_zamani = simdi
                print(f"Manuel sonuc: {okunan}")
            else:
                aktif_durum_text = "Plaka bulunamadi"
                aktif_detay_text = "Kameraya daha yakin tutun"
                aktif_renk = (0, 165, 255)
                aktif_sonuc_zamani = simdi

        elif key == ord('a'):
            oto_mod = not oto_mod
            print(f"Oto Mod: {'ACIK' if oto_mod else 'KAPALI'}")
        elif key == ord('c'):
            bariyer_durum = "KAPALI"
            print("Bariyer manuel kapatildi")
        elif key == ord('l'):
            if os.path.exists(LOG_DOSYASI):
                print("\n--- SON 10 LOG ---")
                with open(LOG_DOSYASI, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-10:]
                    for line in lines:
                        print(line.strip())
            else:
                print("Henuz log yok")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log dosyasi: {LOG_DOSYASI}")

if __name__ == "__main__":
    canli_kamera_akisi_gelismis()
