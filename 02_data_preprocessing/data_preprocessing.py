import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

le = LabelEncoder()
df = pd.read_csv("../01_data_collection/arabam_listings.csv")

# Translate column names from Turkish to English
column_mapping = {
    'İlan No': 'listing_no',
    'İlan Tarihi': 'listing_date',
    'Marka': 'brand',
    'Seri': 'series',
    'Model': 'model',
    'Yıl': 'year',
    'Kilometre': 'mileage',
    'Vites Tipi': 'transmission_type',
    'Yakıt Tipi': 'fuel_type',
    'Kasa Tipi': 'body_type',
    'Renk': 'color',
    'Motor Hacmi': 'engine_size',
    'Motor Gücü': 'engine_power',
    'Çekiş': 'drive_type',
    'Araç Durumu': 'vehicle_status',
    'Ort. Yakıt Tüketimi': 'avg_fuel_consumption',
    'Yakıt Deposu': 'fuel_tank',
    'Boya-değişen': 'paint_changed',
    'Takasa Uygun': 'suitable_for_exchange',
    'Kimden': 'seller_type',
    'Fiyat': 'price',
    'URL': 'url',
    'Ağır Hasarlı': 'heavy_damage'
}

df = df.rename(columns=column_mapping)

def print_column_value_counts(df_in):
    print("Column Names:")
    print(df_in.columns)
    print("\n")

    for column in df_in.columns:
        print(f"Value counts in {column} column:")
        print(df_in[column].value_counts())
        print("\n")

print_column_value_counts(df)

# Filter out specific values
df = df[df["fuel_type"] != "Benzin"]
df = df[df["body_type"] != "Sedan"]

# Drop unnecessary columns
df.drop(
    ["listing_no", "brand", "model", "series",
     "fuel_type", "body_type", "engine_size", "engine_power",
     "drive_type", "vehicle_status", "fuel_tank", "suitable_for_exchange",
     "url"],
    axis=1,
    inplace=True)

df["heavy_damage"] = df["heavy_damage"].fillna("No")

print("\n CLEANED FROM UNNECESSARY COLUMNS \n")
print_column_value_counts(df)
df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v1.csv', index=False)

# Only month information is taken
# No effect on result
df.drop(columns=["listing_date"], inplace=True)

# Age
current_year = 2024
df["age"] = current_year - df["year"]
df.drop(columns=["year"], inplace=True)

# Mileage
df["mileage"] = df["mileage"].str.replace(r'[^\d]', '', regex=True).astype(int)

# Mileage/Age
df["mileage/age"] = (df["mileage"] / df["age"]).astype(int)

# Transmission Type
df = pd.get_dummies(df, columns=["transmission_type"], prefix="transmission_type")

# Color
# No effect on result
df.drop(columns=["color"], inplace=True)

# Avg. Fuel Consumption - Convert to categorical
df["avg_fuel_consumption"] = df["avg_fuel_consumption"].str.replace(r'[^\d.,]', '', regex=True)
df["avg_fuel_consumption"] = df["avg_fuel_consumption"].str.replace(',', '.', regex=False)
df["avg_fuel_consumption"] = pd.to_numeric(df["avg_fuel_consumption"], errors='coerce')
df["avg_fuel_consumption"] = df["avg_fuel_consumption"].round(1).astype(str) + " lt"
df["avg_fuel_consumption"] = le.fit_transform(df["avg_fuel_consumption"])

# Paint-changed
def parse_paint_changed(s):
    """
    Not Specified -> nan changed, nan painted
    All Original -> 0 changed, 0 painted
    <x> painted -> 0 changed, <x> painted
    <x> changed -> <x> changed, 0 painted
    <x> changed <y> painted -> <x> changed, <y> painted
    All Painted -> 0 changed, 13 painted
    """
    if s == 'Belirtilmemiş':
        return pd.Series({'changed': np.nan, 'painted': np.nan})
    elif s == 'Tamamı orjinal':
        return pd.Series({'changed': 0, 'painted': 0})
    elif s == 'Tamamı boyalı':
        return pd.Series({'changed': 0, 'painted': 13})
    else:
        changed = 0
        painted = 0
        parts = s.split(', ')
        for part in parts:
            if 'değişen' in part:
                # Extract number from "<x> changed" format
                changed = int(part.split()[0])
            elif 'boyalı' in part:
                # Extract number from "<y> painted" format
                painted = int(part.split()[0])
        return pd.Series({'changed': changed, 'painted': painted})

df[["changed", "painted"]] = df["paint_changed"].apply(parse_paint_changed)
df["changed"] = df["changed"].astype('Int64')
df["painted"] = df["painted"].astype('Int64')
df = df.drop(["paint_changed"], axis=1)

# Seller Type
df["seller_type"] = le.fit_transform(df["seller_type"])
# No effect on result
df.drop(columns=["seller_type"], inplace=True)

# Price
df["price"] = df["price"].str.replace(r'[^\d]', '', regex=True).astype(int)

# Heavy Damage
df["heavy_damage"] = le.fit_transform(df["heavy_damage"])
# No effect on result
df.drop(columns=["heavy_damage"], inplace=True)

joblib.dump(le, '../02_data_preprocessing/label_encoder.pkl')
print("\nLabelEncoders successfully saved.")

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v2.csv', index=False)
