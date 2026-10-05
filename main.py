import cv2
import easyocr  
from difflib import SequenceMatcher

print("OCR modeli yükleniyor...")
reader = easyocr.Reader(['tr', 'en'], gpu=False)  
print("Model başarıyla yüklendi!")

# 1. Gelişmiş Araç Veritabanı (Sözlük yapısı: Plaka -> Bilgiler)
yetkili_araclar_db = {
    "34ABC123": {"sahip": "Ahmet Yılmaz", "model": "Renault Clio (Gri)"},
    "06KOCA06": {"sahip": "Mehmet Demir", "model": "Toyota Corolla (Beyaz)"},
    "46DZ460":  {"sahip": "Koca Wolf", "model": "Honda Civic (Siyah)"},
    "35IZMIR35": {"sahip": "Ayşe Kaya", "model": "Volkswagen Golf (Kırmızı)"}
}

def plaka_benzerlik_kontrolu(okunan, veritabani, esik_orani=0.80):
    """
    Okunan plaka ile veritabanındaki plakaları esnek bir şekilde kıyaslar.
    Ufak OCR hatalarını (0 ile O karışması vb.) tolere eder.
    """
    en_iyi_eslesme = None
    max_benzerlik = 0.0

    for kayitli_plaka in veritabani.keys():
        # İki metin arasındaki benzerlik oranını hesapla (0.0 ile 1.0 arası)
        oran = SequenceMatcher(None, okunan, kayitli_plaka).ratio()
        if oran > max_benzerlik:
            max_benzerlik = oran
            en_iyi_eslesme = kayitli_plaka

    # Eğer benzerlik oranı belirlenen eşiğin üzerindeyse, eşleşmeyi kabul et
    if max_benzerlik >= esik_orani:
        return en_iyi_eslesme, max_benzerlik
    
    return None, 0.0

def plaka_tanima_islem_akisi(img):
    original = img.copy()
    
    # Ön İşleme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    # Konturları Bulma
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plaka_konturu = None
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            plaka_konturu = approx
            break

    if plaka_konturu is None:
        print("Plaka algılanamadı!")
        return original, None

    # Plaka Alanını Kırpma (ROI)
    x, y, w, h = cv2.boundingRect(plaka_konturu)
    plaka_roi = gray[y:y+h, x:x+w]

    # OCR ile Karakterleri Okuma
    results = reader.readtext(plaka_roi)
    
    okunan_plaka = ""
    for (bbox, text, prob) in results:
        clean_text = "".join(e for e in text if e.isalnum())
        if len(clean_text) >= 5: 
            okunan_plaka = clean_text.upper()
            break

    print(f"OCR'ın Okuduğu Hammadde: {okunan_plaka}")

    # Veritabanı ve Esnek Benzerlik Kontrolü
    eslesen_plaka, benzerlik = plaka_benzerlik_kontrolu(okunan_plaka, yetkili_araclar_db, esik_orani=0.75)

    if eslesen_plaka:
        arac_bilgi = yetkili_araclar_db[eslesen_plaka]
        durum = "YETKILI ARAC - BARIYER ACILDI"
        detay = f"Sahip: {arac_bilgi['sahip']} ({arac_bilgi['model']})"
        renk = (0, 255, 0) # Yeşil
        print(f"Eşleşme Başarılı! Veritabanındaki Kayıt: {eslesen_plaka} (Benzerlik: %{int(benzerlik*100)})")
    else:
        durum = "YETKISIZ ARAC - GIRIS YASAK"
        detay = f"Okunan: {okunan_plaka} (Kayıt Bulunamadı)"
        renk = (0, 0, 255) # Kırmızı
        print("Yetkisiz veya tanınmayan araç!")

    # Görselleştirme
    cv2.drawContours(original, [plaka_konturu], -1, (0, 255, 0), 3)
    cv2.putText(original, f"Plaka: {okunan_plaka}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(original, durum, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
    cv2.putText(original, detay, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return original, plaka_roi

def canli_kamera_akisi():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("\n--- GELİŞMİŞ OTOPARK SİSTEMİ AKTİF ---")
    print("-> Görüntüyü dondurup plaka okutmak için klavyeden **'s'** tuşuna basın.")
    print("-> Sistemden çıkmak için **'q'** tuşuna basın.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Plaka okutmak icin 's' tusuna basin", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Otopark Canli Kamera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("\nKare yakalandı, plaka taranıyor ve veritabanı sorgulanıyor...")
            islenmis_goruntu, roi = plaka_tanima_islem_akisi(frame)
            
            cv2.imshow("Plaka Tanima Sonucu", islenmis_goruntu)
            if roi is not None:
                cv2.imshow("Plaka ROI", roi)
                
            print("Devam etmek için herhangi bir tuşa basın...")
            cv2.waitKey(0)
            cv2.destroyWindow("Plaka Tanima Sonucu")
            try:
                cv2.destroyWindow("Plaka ROI")
            except:
                pass

        elif key == ord('q'):
            print("Sistem kapatılıyor...")
            break

    cap.release()
    cv2.destroyAllWindows()

canli_kamera_akisi()