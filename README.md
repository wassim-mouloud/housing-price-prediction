# Housing Price Prediction Model

A machine learning model to predict housing prices using the California Housing dataset.

## Features

- **Data Preprocessing**: Handles missing values, feature scaling, and encoding
- **Feature Engineering**: Creates new features like rooms_per_household, bedrooms_per_room, and population_per_household
- **Multiple Models**: Compares Linear Regression, Random Forest, and Gradient Boosting
- **Model Evaluation**: Provides comprehensive metrics (RMSE, R², MAE)
- **Visualizations**: Generates plots for model evaluation
- **Model Persistence**: Saves the best model for future use

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the prediction model:
```bash
python housing_price_prediction.py
```

The script will:
1. Load and explore the housing data
2. Preprocess features and engineer new ones
3. Train three different models
4. Compare model performance
5. Save the best model as `best_housing_model.pkl`
6. Generate visualizations as `housing_price_model_evaluation.png`

## Model Performance

The script trains and compares three models:
- **Linear Regression**: Fast baseline model
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble technique

## Output Files

- `best_housing_model.pkl`: Trained model ready for predictions
- `housing_price_model_evaluation.png`: Visualization of model performance

## Dataset

The model uses the California Housing dataset with the following features:
- longitude, latitude
- housing_median_age
- total_rooms, total_bedrooms
- population, households
- median_income
- ocean_proximity
- **Target**: median_house_value
