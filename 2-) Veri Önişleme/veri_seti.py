import pandas as pd

df = pd.read_csv("Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_guncel.csv")

print(df.columns)

print(df["İlan Tarihi"].value_counts())   # 0 = Ağustos, 2 = Eylül, 1 = Ekim
print(df["Vites Tipi"].value_counts())   # 0 = Düz, 1 = Otomatik, 2 = Yarı Otomatik
print(df["Renk"].value_counts())   # 0-11
print(df["Ort. Yakıt Tüketimi"].value_counts())  # 3.8 - 5.0
print(df["Kimden"].value_counts())   # 1 = Sahibinden, 0 = Galeriden
print(df["Ağır Hasarlı"].value_counts())  # 0 = Ağır Hasarlı, 1 = Değil
print(df["Değişen"].value_counts())  # 0-13
print(df["Boyalı"].value_counts())   # 0-13
