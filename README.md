# 🏠 House Price Prediction AI Web App

This is an end-to-end Machine Learning web application built using:

- FastAPI (Backend)
- Scikit-learn (ML Model)
- Kaggle House Price Dataset
- HTML + CSS (Frontend)

## Features
- Uses Gradient Boosting Regressor
- Handles missing values
- Encodes categorical features
- Shows model accuracy (R²)
- Clean professional UI

## How to Run

1. Clone repository
2. Install dependencies:
   pip install -r requirements.txt
3. Train model:
   python train_advanced_model.py
4. Start server:
   uvicorn main:app --reload

## Model Accuracy
R² Score: ~0.90+