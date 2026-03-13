# ============================
# 1. IMPORT LIBRARIES
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# ============================
# 2. LOAD DATA
# ============================

# file_path = "Smart MRP Dummy Data_Draft.xlsx"
file_path = "tata-tech/tata-supply-chain/Smart MRP Dummy Data_Draft.xlsx"
df = pd.read_excel(file_path)

print("Columns in Dataset:")
print(df.columns)


# ============================
# 3. DATA PREPROCESSING
# ============================

# Convert Date column (if exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

# If consumption columns are month-wise like:
# 2025-09, 2025-10, etc → we convert to time-series format

consumption_cols = [col for col in df.columns if "2025" in str(col) or "2026" in str(col)]

# Convert wide to long format
df_long = df.melt(
    id_vars=['Plant', 'Material', 'Mat Type'],
    value_vars=consumption_cols,
    var_name='Month',
    value_name='Demand'
)

df_long['Month'] = pd.to_datetime(df_long['Month'])
df_long = df_long.sort_values('Month')

# Remove null demand
df_long = df_long.dropna(subset=['Demand'])



# ============================
# 4. FEATURE ENGINEERING
# ============================

df_long['Year'] = df_long['Month'].dt.year
df_long['Month_Num'] = df_long['Month'].dt.month
df_long['Quarter'] = df_long['Month'].dt.quarter

# Lag Features
df_long['Lag_1'] = df_long.groupby('Material')['Demand'].shift(1)
df_long['Lag_2'] = df_long.groupby('Material')['Demand'].shift(2)
df_long['Lag_3'] = df_long.groupby('Material')['Demand'].shift(3)

# Rolling Mean
df_long['Rolling_Mean_3'] = df_long.groupby('Material')['Demand'].transform(lambda x: x.rolling(3).mean())

df_long = df_long.dropna()


# ============================
# 5. TRAIN TEST SPLIT
# ============================

features = [
    'Year', 'Month_Num', 'Quarter',
    'Lag_1', 'Lag_2', 'Lag_3',
    'Rolling_Mean_3'
]

X = df_long[features]
y = df_long['Demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# ============================
# 6. TRAIN XGBOOST MODEL
# ============================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


# ============================
# 7. EVALUATION
# ============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== XGBoost Results =====")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)


# ============================
# 8. VISUALIZATION
# ============================

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Demand Forecast vs Actual")
plt.show()


# ============================
# 9. PREDICT NEXT MONTH
# ============================

last_row = df_long.iloc[-1:]

future_input = last_row[features].copy()

future_input['Month_Num'] = future_input['Month_Num'] + 1
if future_input['Month_Num'].values[0] > 12:
    future_input['Month_Num'] = 1
    future_input['Year'] += 1

future_prediction = model.predict(future_input)

print("\nPredicted Demand Next Month:", future_prediction[0])