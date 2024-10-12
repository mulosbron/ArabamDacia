import pandas as pd
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
            print(e)


def main():
    print("Fiyat Tahmin Uygulamasına Hoşgeldiniz!")

    yas = get_valid_input(
        "Araç Yaşı: ",
        int,
        lambda x: x >= 0,
        "Araç yaşı negatif olamaz."
    )

    kilometre = get_valid_input(
        "Kilometre: ",
        int,
        lambda x: x >= 0,
        "Kilometre negatif olamaz."
    )

    ort_yakit_tuketimi = get_valid_input(
        "Ortalama Yakıt Tüketimi (örneğin: 4.0): ",
        float,
        lambda x: 3.0 <= x <= 10.0,
        "Yakıt tüketimi 3 ile 10 arasında olmalıdır."
    )

    vites_tipi = get_valid_input(
        "Vites Tipi (0 = Düz, 1 = Otomatik, 2 = Yarı Otomatik): ",
        int,
        lambda x: x in [0, 1, 2],
        "Vites Tipi sadece 0, 1 veya 2 olabilir."
    )

    degisen = get_valid_input(
        "Değişen Parça Sayısı (örneğin: 0, 1, 2): ",
        int,
        lambda x: x >= 0,
        "Değişen parça sayısı negatif olamaz."
    )

    boyali = get_valid_input(
        "Boyalı Parça Sayısı (örneğin: 0, 1, 2): ",
        int,
        lambda x: x >= 0,
        "Boyalı parça sayısı negatif olamaz."
    )

    model_path = '../3-) Model Eğitme/Makine Öğrenmesi/best_model.pkl'

    if not os.path.exists(model_path):
        print("Model dosyası bulunamadı. Lütfen dosyayı kontrol edin.")
        return

    model = joblib.load(model_path)

    user_input = {
        'yas': yas,
        'kilometre': kilometre,
        'ort_yakit_tuketimi': ort_yakit_tuketimi,
        'vites_tipi_duz': 1 if vites_tipi == 0 else 0,
        'vites_tipi_otomatik': 1 if vites_tipi == 1 else 0,
        'vites_tipi_yari_otomatik': 1 if vites_tipi == 2 else 0,
        'degisen': degisen,
        'boyali': boyali
    }

    df = pd.DataFrame([user_input])

    df['kilometre/yas'] = (df['kilometre'] / df['yas']).astype(int)

    df = df[['yas', 'kilometre', 'ort_yakit_tuketimi', 'kilometre/yas',
             'vites_tipi_duz', 'vites_tipi_otomatik', 'vites_tipi_yari_otomatik',
             'degisen', 'boyali']]

    try:
        fiyat_pred = model.predict(df)[0]
        print(f"\nTahmini Fiyat: {fiyat_pred:,.2f} Türk Lirası")
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {e}")


if __name__ == "__main__":
    main()
