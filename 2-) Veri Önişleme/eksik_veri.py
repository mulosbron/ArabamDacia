import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v2.csv')

missing_columns = df.columns[df.isnull().any()].tolist()
print("Eksik veriye sahip sütunlar:", missing_columns)

models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Bayesian Ridge': BayesianRidge(),
    'SVR': SVR(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

for column in missing_columns:
    print(f"\nŞu anda işlenen sütun: {column}")
    df_missing = df[df[column].isnull()]
    df_not_missing = df[df[column].notnull()]

    if df_missing.empty:
        print(f"{column} sütununda eksik değer bulunmamaktadır.")
        continue

    X = df_not_missing.drop(missing_columns, axis=1)
    y = df_not_missing[column]
    X_missing = df_missing.drop(missing_columns, axis=1)

    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_missing = pd.get_dummies(X_missing, columns=categorical_cols, drop_first=True)

    X_missing = X_missing.reindex(columns=X.columns, fill_value=0)

    best_score = -np.inf
    best_model_name = None
    best_model = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            mean_score = scores.mean()
            print(f"{name} için Ortalama R² Skoru: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"{name} modelinde hata oluştu: {e}")
            continue

    print(f"\nEn iyi model: {best_model_name} - R² Skoru: {best_score:.4f}")

    best_model.fit(X, y)

    predicted_values = best_model.predict(X_missing)

    df.loc[df[column].isnull(), column] = predicted_values

    if column in ['degisen', 'boyali']:
        df[column] = df[column].round().astype(int)
    elif column == 'ort_yakit_tuketimi':
        df[column] = df[column].round(1)

    print(f"{column} sütunundaki eksik değerler {best_model_name} ile dolduruldu.")

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv', index=False)
print("\nEksik değerler dolduruldu ve sonuç 'Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv' dosyasına kaydedildi.")
