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
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v2.csv')

missing_columns = df.columns[df.isnull().any()].tolist()
print("Columns with missing values:", missing_columns)

# Convert avg_fuel_consumption to categorical if it has missing values
if 'avg_fuel_consumption' in missing_columns:
    df['avg_fuel_consumption'] = df['avg_fuel_consumption'].round(1).astype(str) + " lt"
    le = LabelEncoder()
    df['avg_fuel_consumption'] = le.fit_transform(df['avg_fuel_consumption'])

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
    print(f"\nCurrently processing column: {column}")
    df_missing = df[df[column].isnull()]
    df_not_missing = df[df[column].notnull()]

    if df_missing.empty:
        print(f"No missing values found in {column} column.")
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
            print(f"Average R² Score for {name}: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"Error occurred in {name} model: {e}")
            continue

    print(f"\nBest model: {best_model_name} - R² Score: {best_score:.4f}")

    best_model.fit(X, y)
    predicted_values = best_model.predict(X_missing)
    df.loc[df[column].isnull(), column] = predicted_values

    # Round values based on column type
    if column in ['changed', 'painted']:
        df[column] = df[column].round().astype(int)
    elif column == 'avg_fuel_consumption':
        df[column] = df[column].round().astype(int)

    print(f"Missing values in {column} column have been filled using {best_model_name}.")

df.to_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv', index=False)
print("\nMissing values have been filled and results saved to 'Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv'.")
