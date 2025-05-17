import pandas as pd
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv('Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv')

X = df.drop(['price'], axis=1)
y = df['price']

X_encoded = pd.get_dummies(X, drop_first=True)

mi = mutual_info_regression(X_encoded, y, random_state=42)

mi_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Mutual Information': mi
})

mi_df.sort_values('Mutual Information', ascending=False, inplace=True)

print("\nMutual Information of Features with Target Variable:")
print(mi_df)

threshold = 0.03
low_importance_features = mi_df[mi_df['Mutual Information'] < threshold]['Feature'].tolist()

print(f"\nFeatures to be Removed (Mutual Information < {threshold}):")
print(low_importance_features)