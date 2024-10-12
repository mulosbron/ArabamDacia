import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

le = LabelEncoder()
df = pd.read_csv("../1-) Veri Toplama/arabam_ilanlar.csv")


def print_column_value_counts(df):
    print("Sütun İsimleri:")
    print(df.columns)
    print("\n")

    for column in df.columns:
        print(f"{column} sütunundaki değer sayımları:")
        print(df[column].value_counts())
        print("\n")


print_column_value_counts(df)

df = df[df["Yakıt Tipi"] != "Benzin"]
df = df[df["Kasa Tipi"] != "Sedan"]
df.drop(
    ["İlan No", "Marka", "Model", "Seri",
     "Yakıt Tipi", "Kasa Tipi", "Motor Hacmi", "Motor Gücü",
     "Çekiş", "Araç Durumu", "Yakıt Deposu", "Takasa Uygun",
     "URL"],
    axis=1,
    inplace=True)

df["Ağır Hasarlı"] = df["Ağır Hasarlı"].fillna("Hayır")

print("\n GEREKSİZ SÜTUNLARDAN TEMİZLENDİ \n")
print_column_value_counts(df)
df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v1.csv', index=False)

# Sadece ay bilgisi alındı
# Sonuca etkisi yok
df.drop(columns=["İlan Tarihi"], inplace=True)
# df.rename(columns={'İlan Tarihi': 'ilan_tarihi'}, inplace=True)
# df["ilan_tarihi"] = df["ilan_tarihi"].apply(lambda x: x.split()[1])
# df = pd.get_dummies(df, columns=["ilan_tarihi"], prefix='ilan_tarihi')

# Yaş
df.rename(columns={'Yıl': 'yas'}, inplace=True)
simdiki_yil = 2024
df["yas"] = simdiki_yil - df["yas"]

# Kilometre
df.rename(columns={'Kilometre': 'kilometre'}, inplace=True)
df["kilometre"] = df["kilometre"].str.replace(r'[^\d]', '', regex=True).astype(int)

# Kilometre/Yaş
df["kilometre/yas"] = (df["kilometre"] / df["yas"]).astype(int)

# Vites Tipi
df.rename(columns={"Vites Tipi": "vites_tipi"}, inplace=True)
df = pd.get_dummies(df, columns=["vites_tipi"], prefix="vites_tipi")

# Renk
# Sonuca etkisi yok
df.drop(columns=["Renk"], inplace=True)
# df.rename(columns={"Renk": "renk"}, inplace=True)
# df = pd.get_dummies(df, columns=["renk"], prefix="renk")

# Ort. Yakıt Tüketimi
df.rename(columns={"Ort. Yakıt Tüketimi": 'ort_yakit_tuketimi'}, inplace=True)
df["ort_yakit_tuketimi"] = df["ort_yakit_tuketimi"].str.replace(r'[^\d.,]', '', regex=True)
df["ort_yakit_tuketimi"] = df["ort_yakit_tuketimi"].str.replace(',', '.', regex=False)
df["ort_yakit_tuketimi"] = pd.to_numeric(df["ort_yakit_tuketimi"], errors='coerce')


# Boya-değişen
def parse_boya_degis(s):
    """
    Belirtilmemiş -> nan değişen, nan boyalı
    Tamamı orjinal -> 0 değişen, 0 boyalı
    <x> boyalı -> 0 değişen, <x> boyalı
    <x> değişen -> <x> değişen, 0 boyalı
    <x> değişen <y> boyalı -> <x> değişen, <y> boyalı
    Tamamı boyalı-> 0 değişen, 13 boyalı
    """
    if s == 'Belirtilmemiş':
        return pd.Series({'değişen': np.nan, 'boyalı': np.nan})
    elif s == 'Tamamı orjinal':
        return pd.Series({'değişen': 0, 'boyalı': 0})
    elif s == 'Tamamı boyalı':
        return pd.Series({'değişen': 0, 'boyalı': 13})
    else:
        degisen = 0
        boyali = 0
        parts = s.split(', ')
        for part in parts:
            if 'değişen' in part:
                # "<x> değişen" formatından sayıyı çıkarma
                degisen = int(part.split()[0])
            elif 'boyalı' in part:
                # "<y> boyalı" formatından sayıyı çıkarma
                boyali = int(part.split()[0])
        return pd.Series({'değişen': degisen, 'boyalı': boyali})


df[["Değişen", "Boyalı"]] = df["Boya-değişen"].apply(parse_boya_degis)

df["Değişen"] = df["Değişen"].astype('Int64')
df["Boyalı"] = df["Boyalı"].astype('Int64')

df = df.drop(["Boya-değişen"], axis=1)
df.rename(columns={"Değişen": "degisen", "Boyalı": "boyali"}, inplace=True)

# Kimden
df.rename(columns={"Kimden": "kimden"}, inplace=True)
df["kimden"] = le.fit_transform(df["kimden"])
# Sonuca etkisi yok
df.drop(columns=["kimden"], inplace=True)


# Fiyat
df.rename(columns={"Fiyat": "fiyat"}, inplace=True)
df["fiyat"] = df["fiyat"].str.replace(r'[^\d]', '', regex=True).astype(int)

# Ağır Hasarlı
df.rename(columns={"Ağır Hasarlı": "agir_hasarli"}, inplace=True)
df["agir_hasarli"] = le.fit_transform(df["agir_hasarli"])
# Sonuca etkisi yok
df.drop(columns=["agir_hasarli"], inplace=True)


def fix_column_name(col):
    col = col.lower().replace(" ", "_")
    col = col.replace("ç", "c").replace("ğ", "g").replace("ı", "i").replace("ö", "o").replace("ş", "s").replace("ü",
                                                                                                                "u")
    return col


df.columns = [fix_column_name(col) for col in df.columns]

joblib.dump(le, '../2-) Veri Önişleme/label_encoder.pkl')
print("\nLabelEncoder'lar başarıyla kaydedildi.")

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v2.csv', index=False)
