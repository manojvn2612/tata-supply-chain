# Smart MRP — ML Pipeline  v2

## What's in here

| File | Purpose |
|---|---|
| `data_generator.py` | Generates 5 years × 3 materials of synthetic daily MRP data |
| `module_1_lstm_demand_forecast.py` | **PyTorch LSTM** — 14-day multi-step demand forecast |
| `module_2_xgboost_leadtime.py` | **XGBoost** — P50/P90 lead-time prediction per PO |
| `module_3_exception_classifier.py` | **Random Forest** classifier + **Claude API** explanation |
| `module_4_supplier_risk_ml.py` | **Isolation Forest** anomaly detection + **GradientBoosting** late-delivery risk |
| `run_pipeline.py` | Master runner |

## ML model map

```
LSTM           → demand forecasting       (learns trend + multi-cycle seasonality)
XGBoost        → lead-time P50/P90        (quantile regression on PO features)
Random Forest  → exception classification (multi-class, class_weight=balanced)
IsolationForest→ delivery anomaly detect  (unsupervised, no labels needed)
GradientBoosting→ P(late delivery)        (binary, AUC-optimised)
Claude API     → plain-language triage    (LLM explanation of RF prediction)
```

Statistics used **only where ML fails** (cold start, <50 samples):
- If a material has fewer than 50 late-delivery events, GBM is skipped and
  a simple threshold rule is used instead.

## Setup

```bash
pip install -r requirements.txt
```

For Module 3 LLM explanations:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
# Full pipeline (generates data + trains all models + runs inference)
python run_pipeline.py

# Skip retraining (use saved model weights)
python run_pipeline.py --no-retrain

# Single module
python run_pipeline.py --module 1   # LSTM forecast only
python run_pipeline.py --module 2   # XGBoost LT only
python run_pipeline.py --module 3   # RF exceptions only
python run_pipeline.py --module 4   # Supplier risk only
```

## Scaling to real SAP data

Replace `data_generator.py` with a SAP extractor:

```python
# Using pyrfc (requires SAP NW RFC SDK)
pip install pyrfc

from pyrfc import Connection
conn = Connection(ashost='sap-host', sysnr='00', client='100', user='...', passwd='...')

# Pull MSEG for consumption history
result = conn.call('RFC_READ_TABLE', QUERY_TABLE='MSEG',
                   FIELDS=[{'FIELDNAME': f} for f in ['MATNR','BUDAT','MENGE','BWART']],
                   OPTIONS=[{'TEXT': "BWART EQ '261'"}])  # movement type 261 = GI
```

The more SAP data you feed in, the better all 5 ML models become.
Every model is designed to improve automatically as dataset size increases.

## Adding your own data

1. Add rows to `data/training_data.csv` matching the existing column schema
2. Re-run `python run_pipeline.py` — all models retrain automatically
3. Models are saved to `models/` as `.pt` (LSTM) and `.pkl` (sklearn/xgb) files
