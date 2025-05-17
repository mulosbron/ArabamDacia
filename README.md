# ArabamDacia

## Overview
A machine learning-based price prediction system for Dacia Sandero 1.5 dCi Stepway vehicles in the Turkish automotive market. Using real data scraped from Arabam.com, this project tests 23 different regression algorithms and achieves 82.37% R² accuracy with Ridge Regression as the best performing model.

## Purpose
To develop a system that combines web scraping and machine learning techniques for objective vehicle price prediction in the Turkish automotive market. The project aims to provide users with realistic price forecasts while serving as a comprehensive example in the data science field.

## Scope

### Technology Stack:
- **Python**: 3.x
- **Web Scraping**: BeautifulSoup, requests 
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Data Storage**: CSV
- **Model Serialization**: joblib

## Implementation

### Installation:
```bash
# Clone the repository
git clone https://github.com/mulosbron/ArabamDacia.git

# Navigate to project directory
cd ArabamDacia

# Install dependencies
pip install pandas numpy scikit-learn beautifulsoup4 requests joblib python-dotenv
```

### Project Structure:
```
ArabamDacia/
├── 01_data_collection/        # Data Collection
│   ├── data_collection.py     # Web scraping script
│   └── arabam_listings.csv    # Collected data
├── 02_data_preprocessing/     # Data Preprocessing
│   ├── data_preprocessing.py  # Data cleaning script
│   ├── missing_data.py        # Missing value imputation
│   ├── feature_importance.py  # Feature importance analysis
│   └── label_encoder.pkl      # Saved encoders
├── 03_model_training/         # Model Training
│   ├── model_training.py      # Training script
│   └── best_model.pkl         # Saved best model
├── 04_model_testing/          # Model Application
│   └── model_testing.py       # Interactive prediction interface
└── readme.md
```

### Data Pipeline:
1. **Data Collection**: Automatic scraping from Arabam.com
2. **Data Preprocessing**: Cleaning, coding, feature engineering
3. **Lost Value Processing**: Imputation
4. **Model Selection**: Testing regression algorithms
5. **Model Deployment**: Interactive forecasting interface

### Model Performance:
- **Cross-Validation R²**: 0.7479
- **Test Set R²**: 0.7268
- **Best Model**: Ridge Regression
- **Dataset Size**: 317 samples with 9 features

## Screenshots

Results from model comparison showing Ridge Regression as the best performer with 74.79% CV accuracy:

```
Training Set Size: (317, 9)
Test Set Size: (137, 9)

Linear Regression - Cross-Validation R²: 0.7478 (+/- 0.0216)

Ridge Regression - Cross-Validation R²: 0.7479 (+/- 0.0218)

Lasso Regression - Cross-Validation R²: 0.7478 (+/- 0.0216)

Elastic Net - Cross-Validation R²: 0.7367 (+/- 0.0205)

Bayesian Ridge - Cross-Validation R²: 0.5530 (+/- 0.0858)

ARD Regression - Cross-Validation R²: 0.7309 (+/- 0.0271)

Stochastic Gradient Descent - Cross-Validation R²: -32652623867839618089790472192.0000 (+/- 32554726349615014114068791296.0000)

Theil-Sen Regressor - Cross-Validation R²: 0.3781 (+/- 0.1579)

Huber Regressor - Cross-Validation R²: -5.2139 (+/- 7.3843)

Passive Aggressive Regressor - Cross-Validation R²: -53.5695 (+/- 45.2325)

RANSAC Regressor - Cross-Validation R²: 0.7273 (+/- 0.0132)

Orthogonal Matching Pursuit - Cross-Validation R²: 0.3661 (+/- 0.1175)

Support Vector Regressor (RBF Kernel) - Cross-Validation R²: -0.0299 (+/- 0.0400)

Nu Support Vector Regressor - Cross-Validation R²: -0.0322 (+/- 0.0181)

Random Forest Regressor - Cross-Validation R²: 0.6764 (+/- 0.0383)

Gradient Boosting Regressor - Cross-Validation R²: 0.6945 (+/- 0.0359)

AdaBoost Regressor - Cross-Validation R²: 0.6513 (+/- 0.0574)

Extra Trees Regressor - Cross-Validation R²: 0.6445 (+/- 0.0529)

K-Nearest Neighbors Regressor - Cross-Validation R²: 0.4773 (+/- 0.0508)

Decision Tree Regressor - Cross-Validation R²: 0.4167 (+/- 0.0749)

Kernel Ridge Regressor - Cross-Validation R²: 0.5064 (+/- 0.1119)

Gaussian Process Regressor - Cross-Validation R²: -32.4451 (+/- 7.2195)

Multi-layer Perceptron Regressor - Cross-Validation R²: -10.4689 (+/- 2.4140)

Best Model: Ridge Regression with Cross-Validation R²: 0.7479

Ridge Regression Test Set R²: 0.7268

Best model successfully saved: best_model.pkl

Process finished with exit code 0

```