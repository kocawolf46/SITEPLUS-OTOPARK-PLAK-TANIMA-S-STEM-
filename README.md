SITEPLUS PRO v4 - BÜYÜK SİTE OTOPARK SİSTEMİ
KURULUM KILAVUZU
========================================

1) GEREKSİNİMLER
- Python 3.8 veya üzeri
- Webcam / IP Kamera
- İnternet (ilk çalıştırmada EasyOCR modeli indirir ~80MB)

2) KURULUM
Windows / Mac / Linux için aynı:

a) Zip'i bir klasöre çıkar:
   siteplus-pro/

b) Sanal ortam (önerilir):
   python -m venv venv
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

c) Kütüphaneleri kur:
   pip install -r requirements.txt

   Not: easyocr ilk seferde torch indirebilir, biraz sürebilir.

3) ÇALIŞTIRMA
- Ultimate web versiyonu (ÖNERİLEN - Büyük site tasarımı):
  python app_pro.py

  Sonra tarayıcıdan aç:
  http://localhost:5000

  Özellikler:
  - Kamera SÜREKLİ AÇIK, hiç kapanmaz
  - 1.5 sn'de bir otomatik plaka tarar
  - Yetkili -> Bariyer AÇIK (4 sn sonra otomatik kapanır)
  - Fotoğraflar: plates/yetkili/ ve plates/yetkisiz/ klasörüne kaydolur
  - Veritabanı: vehicles.json (web panelden ekle/sil)
  - CSV log indirme

- Basit masaüstü versiyonu:
  python main_v2.py
  Tuşlar: [S] Manuel tara, [A] Oto aç/kapat, [Q] Çıkış, [C] Bariyeri kapat, [L] Log göster

4) TELEGRAM BİLDİRİMİ (Opsiyonel)
app_pro.py dosyasını notepad ile aç, en üstte:

TELEGRAM_BOT_TOKEN = "1234567890:AAH..."  # BotFather'dan al
TELEGRAM_CHAT_ID = "123456789"            # @userinfobot ile öğren
TELEGRAM_AKTIF = True

Nasıl alınır:
1. Telegram'da @BotFather'a gir -> /newbot -> isim ver -> token al
2. Botuna bir mesaj at
3. Tarayıcıda https://api.telegram.org/bot<TOKEN>/getUpdates aç, chat id'yi gör
4. Token ve chat id'yi app_pro.py'ye yapıştır, TELEGRAM_AKTIF = True yap

Sonra:
- Yetkisiz araç gelince: Fotoğrafla alarm gider
- VIP/Yönetici girince: Fotoğrafla giriş bildirimi gider

5) ARAÇ EKLEME
- Web panelden: Sağ alttaki "ARAÇ EKLE" formu
- Veya vehicles.json dosyasını elle düzenle

Örnek vehicles.json:
{
  "46DZ460": {"sahip": "Koca Wolf", "model": "Honda Civic (Siyah)", "daire": "C-01", "tip": "VIP", "tel": ""}
}

6) SORUN GİDERME
- Kamera açılmıyor: app_pro.py içinde cv2.VideoCapture(0) -> 1 yap, veya 2 dene
- FPS düşük: app_pro.py'de AUTO_SCAN_INTERVAL = 1.5'i 2.0 yap
- OCR okumuyor: Plakayı kameraya daha yakın tut, aydınlık olsun, plates/ klasöründeki foto net mi kontrol et
- Port 5000 dolu: app.py son satırda port=5000'i 5001 yap

7) DOSYA LİSTESİ
app_pro.py      -> Ultimate web sistemi (FLASK + Telegram + Foto + JSON) - BUNU KULLAN
app.py          -> Web sistemi v3 (Telegram yok)
main_v2.py      -> Masaüstü gelişmiş versiyon (OpenCV penceresi, kamera sürekli açık)
main.py         -> İlk basit versiyon
vehicles.json   -> Araç veritabanı
requirements.txt-> Gerekli kütüphaneler
kurulum.txt     -> Bu dosya
plates/         -> Çalışınca otomatik oluşur, fotoğraflar buraya kaydolur

8) İLETİŞİM
Geliştirici: Ali Koca - Koca Residence
Sistem: SITEPLUS PRO v4.0 ULTIMATE

İyi kullanımlar!
