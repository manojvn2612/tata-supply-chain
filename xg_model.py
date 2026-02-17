# ==========================================
# SUPPLY CHAIN DEMAND FORECASTING
# XGBoost Model with Full Evaluation
# ==========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

from xgboost import XGBRegressor

# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "demand_forecasting_dataset (1).csv"
df = pd.read_csv(DATA_PATH)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

df['date'] = pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df.drop(columns=['date'], inplace=True)

# ==========================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==========================================

cat_cols = ['product_id', 'category_id', 'store_id']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==========================================
# 4. DEFINE FEATURES & TARGET
# ==========================================

X = df.drop(columns=['target_demand'])
y = df['target_demand']

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 6. TRAIN XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 7. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 8. EVALUATION METRICS
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

# Adjusted R2
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# MAPE (avoid divide by zero)
mape = np.mean(
    np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))
) * 100

# ==========================================
# 9. PRINT RESULTS
# ==========================================

print("\n========== XGBOOST MODEL PERFORMANCE ==========")
print(f"MAE (Mean Absolute Error)        : {mae:.4f}")
print(f"MSE (Mean Squared Error)         : {mse:.4f}")
print(f"RMSE (Root Mean Squared Error)   : {rmse:.4f}")
print(f"R2 Score                         : {r2:.4f}")
print(f"Adjusted R2                      : {adj_r2:.4f}")
print(f"MAPE (%)                         : {mape:.4f}")
print(f"Explained Variance Score         : {explained_var:.4f}")

# ==========================================
# 10. SAVE MODEL
# ==========================================

with open("demand_xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nModel saved successfully!")

# ==========================================
# 11. PREDICTION ON X_test
# ==========================================
print("\nPredictions on X_test:")
print(y_pred)

# ==========================================
# 12. PRINT y_test vs y_pred TABULAR
# ==========================================
import pandas as pd
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nActual vs Predicted (Tabular):")
print(results_df.head(20).to_string(index=False))  # Show first 20 rows
