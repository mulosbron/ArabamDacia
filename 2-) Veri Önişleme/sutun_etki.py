import pandas as pd
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv('../2-) Veri Önişleme/Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv')

X = df.drop(['fiyat'], axis=1)
y = df['fiyat']

X_encoded = pd.get_dummies(X, drop_first=True)

mi = mutual_info_regression(X_encoded, y, random_state=42)

mi_df = pd.DataFrame({
    'Özellik': X_encoded.columns,
    'Karşılıklı Bilgi': mi
})

mi_df.sort_values('Karşılıklı Bilgi', ascending=False, inplace=True)

print("\nÖzelliklerin Hedef Değişkenle Olan Karşılıklı Bilgisi:")
print(mi_df)

threshold = 0.03
low_importance_features = mi_df[mi_df['Karşılıklı Bilgi'] < threshold]['Özellik'].tolist()

print(f"\nÇıkarılacak Özellikler (Karşılıklı Bilgi < {threshold}):")
print(low_importance_features)