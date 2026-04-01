import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ======================================
# 1. Reproducibility
# ======================================
np.random.seed(42)
torch.manual_seed(42)

# ======================================
# 2. Generate 36-Month Synthetic Demand
# ======================================
months = pd.date_range(start="2023-01-01", periods=36, freq="MS")

trend = np.linspace(100, 160, 36)
seasonality = 15 * np.sin(np.arange(36) * 2 * np.pi / 12)
noise = np.random.normal(0, 3, 36)

demand = trend + seasonality + noise

df = pd.DataFrame({
    "Month": months,
    "Demand": demand
})

# ======================================
# 3. Feature Engineering (Lag Features)
# ======================================
for lag in range(1, 4):
    df[f"Lag_{lag}"] = df["Demand"].shift(lag)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ======================================
# 4. Scaling (Only Numeric Columns)
# ======================================
scaler = MinMaxScaler()

numeric_cols = ["Demand", "Lag_1", "Lag_2", "Lag_3"]
scaled = scaler.fit_transform(df[numeric_cols])

# ======================================
# 5. Create Sequences
# ======================================
sequence_length = 3
X = []
y = []

for i in range(len(scaled) - sequence_length):
    X.append(scaled[i:i+sequence_length])
    y.append(scaled[i+sequence_length][0])

X = np.array(X)
y = np.array(y)

# Train-test split (time-based)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================================
# 6. Define LSTM Model
# ======================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=4, hidden_size=32, num_layers=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ======================================
# 7. Training
# ======================================
epochs = 100

for epoch in range(epochs):
    model.train()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

# ======================================
# 8. Evaluation
# ======================================
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

# Inverse scaling
pred_vals = []
actual_vals = []

for i in range(len(predictions)):
    temp = np.zeros((1, 4))
    temp[0][0] = predictions[i].item()
    inv = scaler.inverse_transform(temp)
    pred_vals.append(inv[0][0])

    temp2 = np.zeros((1, 4))
    temp2[0][0] = y_test[i].item()
    inv2 = scaler.inverse_transform(temp2)
    actual_vals.append(inv2[0][0])

# ======================================
# 9. Metrics
# ======================================
rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
mae = mean_absolute_error(actual_vals, pred_vals)
r2 = r2_score(actual_vals, pred_vals)

print("\nModel Performance:")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))
print("R2  :", round(r2, 4))

# ======================================
# 10. Plot Forecast
# ======================================
plt.figure()
plt.plot(actual_vals)
plt.plot(pred_vals)
plt.title("LSTM Demand Forecast vs Actual")
plt.xlabel("Test Time Steps")
plt.ylabel("Demand")
plt.show()

print("\nPipeline completed successfully.")