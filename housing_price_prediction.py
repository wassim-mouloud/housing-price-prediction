"""
Housing Price Prediction Model (Optimized)
This script builds and evaluates a machine learning model to predict housing prices
using the California Housing dataset. 

Adjustments made:
1. Removed capped data (>= $500,000) to improve accuracy.
2. Tuned Random Forest hyperparameters to prevent overfitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("=" * 80)
print("HOUSING PRICE PREDICTION MODEL (OPTIMIZED)")
print("=" * 80)

df = pd.read_csv('housing.csv')

print("\n1. DATA OVERVIEW")
print("-" * 80)
print(f"Original Dataset shape: {df.shape}")


print("\n[ADJUSTMENT] Filtering out capped values ($500,001)...")
df = df[df['median_house_value'] < 500000]
print(f"New Dataset shape: {df.shape}")

print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n\n2. DATA PREPROCESSING")
print("-" * 80)

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                     'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n\n3. FEATURE ENGINEERING")
print("-" * 80)

X['rooms_per_household'] = X['total_rooms'] / X['households']
X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
X['population_per_household'] = X['population'] / X['households']

numerical_features.extend(['rooms_per_household', 'bedrooms_per_room', 'population_per_household'])

print("Created new features: rooms_per_household, bedrooms_per_room, population_per_household")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================

print("\n\n4. TRAIN-TEST SPLIT")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# ============================================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================================

print("\n\n5. CREATING PREPROCESSING PIPELINE")
print("-" * 80)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================

print("\n\n6. MODEL TRAINING")
print("-" * 80)


models = {
    'Linear Regression': LinearRegression(),
    
    'Random Forest (Tuned)': RandomForestRegressor(
        n_estimators=200,      
        max_depth=20,         
        min_samples_split=5,   
        min_samples_leaf=4,    
        random_state=42, 
        n_jobs=-1
    ),
    
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=150, 
        max_depth=5,           
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    full_pipeline.fit(X_train, y_train)

    y_train_pred = full_pipeline.predict(X_train)
    y_test_pred = full_pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    results[name] = {
        'model': full_pipeline,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'predictions': y_test_pred
    }

    print(f"  Train RMSE: ${train_rmse:,.2f}")
    print(f"  Test RMSE: ${test_rmse:,.2f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: ${test_mae:,.2f}")

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================

print("\n\n7. MODEL COMPARISON")
print("-" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train RMSE': [results[m]['train_rmse'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()]
})

print(comparison_df.to_string(index=False))

best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================

if 'Forest' in best_model_name or 'Boosting' in best_model_name:
    print("\n\n8. FEATURE IMPORTANCE")
    print("-" * 80)

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)

    feature_importances = best_model.named_steps['model'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

# ============================================================================
# 9. SAMPLE PREDICTIONS
# ============================================================================

print("\n\n9. SAMPLE PREDICTIONS")
print("-" * 80)

sample_indices = np.random.choice(len(y_test), 10, replace=False)
predictions_df = pd.DataFrame({
    'Actual Price': y_test.iloc[sample_indices].values,
    'Predicted Price': results[best_model_name]['predictions'][sample_indices],
    'Difference': y_test.iloc[sample_indices].values - results[best_model_name]['predictions'][sample_indices]
})

print(predictions_df.to_string(index=False))

# ============================================================================
# 10. SAVE THE BEST MODEL
# ============================================================================

print("\n\n10. SAVING MODEL")
print("-" * 80)

import pickle

with open('best_housing_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model saved to 'best_housing_model.pkl'")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

print("\n\n11. CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].scatter(y_test, results[best_model_name]['predictions'], alpha=0.5, color='royalblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title(f'Actual vs Predicted ({best_model_name})')
axes[0, 0].grid(True, alpha=0.3)

residuals = y_test - results[best_model_name]['predictions']
axes[0, 1].scatter(results[best_model_name]['predictions'], residuals, alpha=0.5, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot (Homoscedasticity Check)')
axes[0, 1].grid(True, alpha=0.3)

model_names = list(results.keys())
test_rmses = [results[m]['test_rmse'] for m in model_names]
axes[1, 0].bar(model_names, test_rmses, color=['skyblue', 'lightgreen', 'orange'])
axes[1, 0].set_ylabel('RMSE ($)')
axes[1, 0].set_title('Model Comparison - Test RMSE (Lower is Better)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Prediction Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('housing_price_model_evaluation.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'housing_price_model_evaluation.png'")

plt.show()

print("\n" + "=" * 80)
print("OPTIMIZED TRAINING COMPLETE!")
print("=" * 80)