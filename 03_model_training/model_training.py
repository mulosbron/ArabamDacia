import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression,
    SGDRegressor, TheilSenRegressor, HuberRegressor, PassiveAggressiveRegressor,
    RANSACRegressor, OrthogonalMatchingPursuit
)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('../02_data_preprocessing/Dacia_1-5_dCi_Stepway_Sandero_Hatchback5_v3.csv')

X = data.drop(['price'], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Bayesian Ridge': BayesianRidge(),
    'ARD Regression': ARDRegression(),
    'Stochastic Gradient Descent': SGDRegressor(max_iter=1000, tol=1e-3),
    'Theil-Sen Regressor': TheilSenRegressor(),
    'Huber Regressor': HuberRegressor(max_iter=1000, tol=1e-3),
    'Passive Aggressive Regressor': PassiveAggressiveRegressor(),
    'RANSAC Regressor': RANSACRegressor(),
    'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
    'Support Vector Regressor (RBF Kernel)': SVR(kernel='rbf'),
    'Nu Support Vector Regressor': NuSVR(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
    'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Kernel Ridge Regressor': KernelRidge(),
    'Gaussian Process Regressor': GaussianProcessRegressor(),
    'Multi-layer Perceptron Regressor': MLPRegressor(random_state=42, max_iter=1000)
}

model_performance = {}

best_model_name = None
best_score = -np.inf
best_model = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1
        )
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        model_performance[name] = {'R2 Mean': mean_score, 'R2 Std': std_score}
        print(f"\n{name} - Cross-Validation R²: {mean_score:.4f} (+/- {std_score:.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = model
    except Exception as e:
        print(f"\nError occurred during {name} model training: {e}")
        continue

if best_model_name is None:
    raise Exception("No model was successfully trained. Please review your dataset and model selections.")
print(f"\nBest Model: {best_model_name} with Cross-Validation R²: {best_score:.4f}")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n{best_model_name} Test Set R²: {r2:.4f}")

model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)

print(f"\nBest model successfully saved: {model_filename}")
