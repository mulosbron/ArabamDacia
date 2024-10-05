import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5.csv')

missing_columns = df.columns[df.isnull().any()].tolist()

for column in missing_columns:
    print(f"\nŞu anda işlenen sütun: {column}")
    df_missing = df[df[column].isnull()]
    df_not_missing = df[df[column].notnull()]
    X = df_not_missing.drop([column], axis=1)
    y = df_not_missing[column]
    X_missing = df_missing.drop([column], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X_missing = pd.get_dummies(X_missing, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"{column} için Test Seti RMSE: {rmse:.2f}")
    if not df_missing.empty:
        X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)
        predicted_values = model.predict(X_missing)
        df.loc[df[column].isnull(), column] = predicted_values
        df[column] = df[column].round(1)
        print(f"{column} sütunundaki eksik değerler tahmin edildi ve dolduruldu.")
    else:
        print(f"{column} sütununda eksik değer bulunmamaktadır.")

df["Boyalı"] = df["Boyalı"].astype(int)
df["Değişen"] = df["Değişen"].astype(int)
print("\nEksik Değer Sayısı:")
print(df.isnull().sum())

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_guncel.csv', index=False)
