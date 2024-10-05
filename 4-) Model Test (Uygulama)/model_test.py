import pandas as pd
import numpy as np
import joblib
import os

def get_valid_input(prompt, type_func, condition, error_msg):
    while True:
        try:
            value = type_func(input(prompt))
            if not condition(value):
                print(error_msg)
                continue
            return value
        except Exception as e:
            print("Geçersiz giriş. Lütfen doğru türde bir değer giriniz.")

def main():
    print("Fiyat Tahmin Uygulamasına Hoşgeldiniz!")

    ilan_tarihi = get_valid_input(
        "İlan Tarihi (0 = Ağustos, 2 = Eylül, 1 = Ekim) ",
        int,
        lambda x: x >= 0,
        "Ay 0'dan küçük olamaz."
    )

    kilometre_age = get_valid_input(
        "Kilometre/Yaş (0 veya daha büyük bir tam sayı): ",
        int,
        lambda x: x >= 0,
        "Kilometre/Yaş 0'dan küçük olamaz."
    )

    vites_tipi = get_valid_input(
        "Vites Tipi (0 = Düz, 1 = Otomatik, 2 = Yarı Otomatik): ",
        int,
        lambda x: x in [0, 1, 2],
        "Vites Tipi sadece 0, 1 veya 2 olabilir."
    )

    valid_renk = list(range(0, 12))
    renk = get_valid_input(
        f"Renk (0 ile 11 arasında bir sayı, mevcut renkler {valid_renk}): ",
        int,
        lambda x: x in valid_renk,
        f"Renk sadece {valid_renk} değerlerinden biri olabilir."
    )

    ort_yakit_tuketimi = get_valid_input(
        "Ortalama Yakıt Tüketimi (3.8 ile 5 arasında bir değer): ",
        float,
        lambda x: 3.8 <= x <= 5.0,
        "Ortalama Yakıt Tüketimi 3.8 ile 5 arasında olmalıdır."
    )

    kimden = get_valid_input(
        "Kimden (0 = Galeriden, 1 = Sahibinden): ",
        int,
        lambda x: x in [0, 1],
        "Kimden sadece 0 veya 1 olabilir."
    )

    agir_hasarli = get_valid_input(
        "Araç ağır hasarlı mı? (1 = Hasarsız, 0 = Hasarlı): ",
        int,
        lambda x: x in [0, 1],
        "Ağır Hasarlı sadece 0 veya 1 olabilir."
    )

    degisen = get_valid_input(
        "Değişen (0 veya daha büyük bir tam sayı): ",
        int,
        lambda x: x >= 0,
        "Değişen 0'dan küçük olamaz."
    )

    boyali = get_valid_input(
        "Boyalı (0 veya daha büyük bir tam sayı): ",
        int,
        lambda x: x >= 0,
        "Boyalı 0'dan küçük olamaz."
    )

    pipeline_path = '../3-) Model Eğitme/best_model.pkl'
    encoder_path = '../3-) Model Eğitme/encoder.pkl'
    scaler_path = '../3-) Model Eğitme/scaler.pkl'

    # Model, Encoder ve Scaler dosyalarının varlığını kontrol etme
    if not os.path.exists(pipeline_path):
        print("Model dosyası bulunamadı. Lütfen önce modeli eğitin ve kaydedin.")
        return
    if not os.path.exists(encoder_path):
        print("Encoder dosyası bulunamadı. Lütfen önce modeli eğitin ve kaydedin.")
        return
    if not os.path.exists(scaler_path):
        print("Scaler dosyası bulunamadı. Lütfen önce modeli eğitin ve kaydedin.")
        return

    # Model, Encoder ve Scaler'ı Yükleme
    try:
        model = joblib.load(pipeline_path)
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Dosyalar yüklenirken bir hata oluştu: {e}")
        return

    # Kullanıcı Girdisini DataFrame'e Dönüştürme
    input_data = {
        'İlan Tarihi': [ilan_tarihi],
        'Vites Tipi': [vites_tipi],
        'Renk': [renk],
        'Kimden': [kimden],
        'Ağır Hasarlı': [agir_hasarli],
        'Kilometre/Yaş': [kilometre_age],
        'Ort. Yakıt Tüketimi': [ort_yakit_tuketimi],
        'Değişen': [degisen],
        'Boyalı': [boyali]
    }

    input_df = pd.DataFrame(input_data)

    print("\nTahmin için kullanılan girdi verileri:")
    print(input_df)

    # Ön İşleme Adımları
    try:
        # Kategorik Özelliklerin Encode Edilmesi
        categorical_features = ['İlan Tarihi', 'Vites Tipi', 'Renk', 'Kimden', 'Ağır Hasarlı']
        numerical_features = ['Kilometre/Yaş', 'Ort. Yakıt Tüketimi', 'Değişen', 'Boyalı']

        X_cat = encoder.transform(input_df[categorical_features])
        X_num = scaler.transform(input_df[numerical_features])

        # Ön İşlemeli Verilerin Birleştirilmesi
        X_processed = np.hstack([X_num, X_cat])

    except Exception as e:
        print(f"Veri ön işleme sırasında bir hata oluştu: {e}")
        return

    # Tahmin Yapma
    try:
        fiyat_pred = model.predict(X_processed)[0]
        print(f"\nTahmini Fiyat: {fiyat_pred:,.2f} DOLAR")
    except Exception as e:
        print(f"Tahmin yapılırken bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
