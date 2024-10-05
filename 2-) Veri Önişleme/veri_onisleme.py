import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../1-) Veri Toplama/arabam_ilanlar.csv")
print(df.columns)

df = df.drop(
    ["Araç Durumu", "Çekiş", "Yakıt Deposu", "Yakıt Tipi", "URL", "İlan No", "Motor Hacmi",
     "Takasa Uygun"],
    axis=1)
df["Araba"] = df["Marka"] + " " + df["Model"] + " " + df["Seri"] + " " + df["Kasa Tipi"]
df = df.drop(["Kasa Tipi", "Marka", "Model", "Seri"], axis=1)
df["Ağır Hasarlı"] = df["Ağır Hasarlı"].fillna("Hayır")
print(df.columns)
print(df["Boya-değişen"].value_counts())


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


df[['Değişen', 'Boyalı']] = df['Boya-değişen'].apply(parse_boya_degis)

df['Değişen'] = df['Değişen'].astype('Int64')
df['Boyalı'] = df['Boyalı'].astype('Int64')

df = df.drop(["Araba", "Boya-değişen", "Motor Gücü"], axis=1)

df['İlan Tarihi'] = df['İlan Tarihi'].apply(lambda x: x.split()[1])

le = LabelEncoder()

df['Vites Tipi'] = le.fit_transform(df['Vites Tipi'])
df['Renk'] = le.fit_transform(df['Renk'])
df['Kimden'] = le.fit_transform(df['Kimden'])
df['Ağır Hasarlı'] = le.fit_transform(df['Ağır Hasarlı'])
df['İlan Tarihi'] = le.fit_transform(df['İlan Tarihi'])

df['Kilometre'] = df['Kilometre'].str.replace(r'[^\d]', '', regex=True).astype(int)
df['Fiyat'] = df['Fiyat'].str.replace(r'[^\d]', '', regex=True).astype(int)
df['Fiyat'] = (df['Fiyat'] // 34.17).astype(int)
df['Ort. Yakıt Tüketimi'] = df['Ort. Yakıt Tüketimi'].str.replace(r'[^\d.,]', '', regex=True)
df['Ort. Yakıt Tüketimi'] = df['Ort. Yakıt Tüketimi'].str.replace(',', '.', regex=False)
df['Ort. Yakıt Tüketimi'] = pd.to_numeric(df['Ort. Yakıt Tüketimi'], errors='coerce')

current_year = 2024
df.rename(columns={'Yıl': 'Yaş'}, inplace=True)
df['Yaş'] = current_year - df['Yaş']
df['Yaş'] = (df['Kilometre'] / df['Yaş']).astype(int)
df.rename(columns={'Yaş': 'Kilometre/Yaş'}, inplace=True)
df.drop('Kilometre', axis=1, inplace=True)
df = df[[col for col in df.columns if col != 'Fiyat'] + ['Fiyat']]

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5.csv', index=False)
