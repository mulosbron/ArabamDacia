import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import pandas as pd
import time

load_dotenv()

headers = {
    "User-Agent": os.getenv("USER_AGENT")
}
base_url = os.getenv("BASE_URL")
url_path = os.getenv("URL_PATH")
list_url = f"{base_url}{url_path}"
ilanlar = []

for page in range(1, 7):
    print(f"{page}. sayfa işleniyor...")
    sayfa_url = f"{list_url}?page={page}"
    response = requests.get(sayfa_url, headers=headers)
    if response.status_code != 200:
        print(f"{page}. sayfa alınamadı. Durum Kodu: {response.status_code}")
        continue
    soup = BeautifulSoup(response.content, "html.parser")
    ilan_listesi = soup.find_all("tr", class_="listing-list-item")
    for ilan in ilan_listesi:
        a_tag = ilan.find("a", href=True)
        if a_tag:
            ilan_url = base_url + a_tag['href']
            ilanlar.append(ilan_url)
    time.sleep(2)
print(f"Toplam {len(ilanlar)} ilan bulundu.")


def ilan_detaylarini_getir(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"URL alınamadı: {url}")
        return None
    soup = BeautifulSoup(response.content, "html.parser")
    detaylar = {}
    detaylar_div = soup.find("div", class_="product-properties-details")
    if not detaylar_div:
        print(f"Detaylar bulunamadı: {url}")
        return None
    ozellikler = detaylar_div.find_all("div", class_="property-item")
    for item in ozellikler:
        anahtar_div = item.find("div", class_="property-key")
        deger_div = item.find("div", class_="property-value")
        if anahtar_div and deger_div:
            anahtar = anahtar_div.get_text(strip=True).replace(":", "")
            deger = deger_div.get_text(strip=True)
            detaylar[anahtar] = deger
    fiyat_kapsayici = soup.find("div", class_="product-price-container")
    if fiyat_kapsayici:
        fiyat_div = fiyat_kapsayici.find("div", {"data-testid": "desktop-information-price"})
        if fiyat_div:
            fiyat = fiyat_div.get_text(strip=True)
            detaylar["Fiyat"] = fiyat
        else:
            print(f"Fiyat bilgisi kapsayıcıda bulunamadı: {url}")
            detaylar["Fiyat"] = "Bilinmiyor"
    else:
        print(f"Fiyat kapsayıcı bulunamadı: {url}")
        detaylar["Fiyat"] = "Bilinmiyor"
    detaylar["URL"] = url
    return detaylar


detayli_ilanlar = []
for idx, ilan_url in enumerate(ilanlar, 1):
    print(f"{idx}. İlan işleniyor: {ilan_url}")
    detaylar = ilan_detaylarini_getir(ilan_url)
    if detaylar:
        detayli_ilanlar.append(detaylar)
    time.sleep(2)

df = pd.DataFrame(detayli_ilanlar)
df.to_csv("arabam_ilanlar.csv", index=False, encoding='utf-8-sig')
print("Veriler başarıyla 'arabam_ilanlar.csv' dosyasına kaydedildi.")
