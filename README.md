# ArabamDacia

## Overview
A machine learning-based price prediction system for Dacia Sandero 1.5 dCi Stepway vehicles in the Turkish automotive market. Using real data scraped from Arabam.com, this project tests 23 different regression algorithms and achieves 82.37% R² accuracy with Ridge Regression as the best performing model.

## Purpose
To develop a system that combines web scraping and machine learning techniques for objective vehicle price prediction in the Turkish automotive market. The project aims to provide users with realistic price forecasts while serving as a comprehensive example in the data science field.

### Key Objectives:
- Create an automated data collection and updatable data pipeline
- Intelligently fill missing data and clean the dataset
- Perform comprehensive comparison to select the optimal regression model

## Scope

### Technology Stack:
- **Python**: 3.x
- **Web Scraping**: BeautifulSoup, requests 
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Data Storage**: CSV
- **Model Serialization**: joblib

### Project Features:
- Automated data collection via web scraping
- Missing value imputation using 12 different algorithms
- Comparative analysis of 23 regression algorithms
- Ridge Regression achieving 79.68% test accuracy
- Interactive price prediction interface

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

### Usage:
```bash
# Run the interactive prediction tool
cd "04_model_test"
python model_test.py
```

### Data Pipeline:
1. **Data Collection**: Automated scraping from Arabam.com
2. **Data Preprocessing**: Cleaning, encoding, feature engineering
3. **Missing Value Handling**: Intelligent imputation using 12 algorithms
4. **Model Selection**: Testing 23 regression algorithms
5. **Model Deployment**: Interactive prediction interface

### Model Performance:
- **Cross-Validation R²**: 0.8237
- **Test Set R²**: 0.7968
- **Best Model**: Ridge Regression
- **Dataset Size**: 338 samples with 9 features

## Screenshots

Results from model comparison showing Ridge Regression as the best performer with 82.37% CV accuracy:

```
Training Set Size: (236, 9)
Test Set Size: (102, 9)

Linear Regression - Cross-Validation R²: 0.8237 (+/- 0.0303)

Ridge Regression - Cross-Validation R²: 0.8237 (+/- 0.0303)

Lasso Regression - Cross-Validation R²: 0.8237 (+/- 0.0303)

Elastic Net - Cross-Validation R²: 0.8023 (+/- 0.0337)

Bayesian Ridge - Cross-Validation R²: 0.6178 (+/- 0.1003)

ARD Regression - Cross-Validation R²: 0.8109 (+/- 0.0364)

Stochastic Gradient Descent - Cross-Validation R²: -16747615789034279199064457216.0000 (+/- 15406528372484856481497219072.0000)

Theil-Sen Regressor - Cross-Validation R²: 0.5143 (+/- 0.0611)

Huber Regressor - Cross-Validation R²: -4.8779 (+/- 6.5635)

Passive Aggressive Regressor - Cross-Validation R²: -20.8927 (+/- 10.6177)

RANSAC Regressor - Cross-Validation R²: 0.7955 (+/- 0.0335)

Orthogonal Matching Pursuit - Cross-Validation R²: 0.4550 (+/- 0.0607)

Support Vector Regressor (RBF Kernel) - Cross-Validation R²: -0.0289 (+/- 0.0287)

Nu Support Vector Regressor - Cross-Validation R²: -0.0264 (+/- 0.0197)

Random Forest Regressor - Cross-Validation R²: 0.7785 (+/- 0.0262)

Gradient Boosting Regressor - Cross-Validation R²: 0.7892 (+/- 0.0311)

AdaBoost Regressor - Cross-Validation R²: 0.7580 (+/- 0.0395)

Extra Trees Regressor - Cross-Validation R²: 0.7763 (+/- 0.0257)

K-Nearest Neighbors Regressor - Cross-Validation R²: 0.5667 (+/- 0.1127)

Decision Tree Regressor - Cross-Validation R²: 0.5963 (+/- 0.0533)

Kernel Ridge Regressor - Cross-Validation R²: 0.5721 (+/- 0.0432)

Gaussian Process Regressor - Cross-Validation R²: -30.9341 (+/- 3.0360)

Multi-layer Perceptron Regressor - Cross-Validation R²: -9.4722 (+/- 2.8524)

Best Model: Ridge Regression with Cross-Validation R²: 0.8237

Ridge Regression Test Set R²: 0.7968

Best model successfully saved: best_model.pkl
```


