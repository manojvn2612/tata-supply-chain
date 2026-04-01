# """
# module_4_supplier_risk_ml.py
# ------------------------------
# ML Module 4: Supplier Risk Scoring  (Isolation Forest + GradientBoosting)

# Two ML models:
#   1. Isolation Forest (unsupervised)
#      Detects anomalous delivery behaviour — no labels needed.
#      Flags shipments that are statistical outliers vs. the supplier's normal pattern.
#      Output: anomaly score per delivery event.

#   2. GradientBoostingClassifier (supervised)
#      Predicts probability that next delivery will be LATE (binary).
#      Trained on historical on-time/late labels derived from lt_deviation_days.
#      Output: P(late), risk tier (GREEN / AMBER / RED).

# Why ML over weighted scoring:
#   - Isolation Forest catches new risk patterns not encoded in any rule
#   - GBM learns interaction effects between seasonal timing, PO size, and delays
#   - Both models update automatically as more data arrives

# Run:  python module_4_supplier_risk_ml.py
# """

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, classification_report
# import joblib
# from pathlib import Path

# MODELS_DIR = Path("models")
# MODELS_DIR.mkdir(exist_ok=True)

# # ── Feature sets ───────────────────────────────────────────────────────────────

# ANOMALY_FEATURES = [
#     "actual_lead_time_days",
#     "lt_deviation_days",
#     "transit_delay_days",
#     "po_slippage_days",
#     "qi_rejection",
#     "stock_cover_days",
# ]

# RISK_FEATURES = [
#     "nominal_lead_time_days",
#     "transit_risk_level",
#     "po_slippage_days",
#     "qi_rejection",
#     "month",
#     "quarter",
#     "month_sin",
#     "month_cos",
#     "demand_rolling_slope",
#     "daily_demand",
#     "stock_cover_days",
#     "lt_deviation_days",
# ]

# RISK_THRESHOLDS = {"RED": 0.55, "AMBER": 0.30}   # P(late) thresholds


# # ── Model 1: Isolation Forest ─────────────────────────────────────────────────

# def train_isolation_forest(mat_df: pd.DataFrame, mat_id: str) -> IsolationForest:
#     print(f"    Training Isolation Forest for {mat_id}...", end=" ")
#     X = mat_df[ANOMALY_FEATURES].values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     iso = IsolationForest(
#         n_estimators=200,
#         contamination=0.05,   # expect ~5% anomalous deliveries
#         random_state=42,
#         n_jobs=-1,
#     )
#     iso.fit(X_scaled)

#     scores      = iso.decision_function(X_scaled)
#     n_anomalies = (iso.predict(X_scaled) == -1).sum()
#     print(f"done  |  anomalies detected: {n_anomalies} ({n_anomalies/len(X_scaled):.1%})")

#     safe_id = mat_id.replace("-", "_")
#     joblib.dump(iso,    MODELS_DIR / f"iso_forest_{safe_id}.pkl")
#     joblib.dump(scaler, MODELS_DIR / f"iso_scaler_{safe_id}.pkl")
#     return iso, scaler


# def detect_anomalies(mat_id: str, feature_rows: np.ndarray, iso=None, scaler=None) -> np.ndarray:
#     if iso is None:
#         safe_id = mat_id.replace("-", "_")
#         iso     = joblib.load(MODELS_DIR / f"iso_forest_{safe_id}.pkl")
#         scaler  = joblib.load(MODELS_DIR / f"iso_scaler_{safe_id}.pkl")
#     X_scaled = scaler.transform(feature_rows)
#     scores   = iso.decision_function(X_scaled)   # lower = more anomalous
#     labels   = iso.predict(X_scaled)              # -1 = anomaly, 1 = normal
#     return scores, labels


# # ── Model 2: GradientBoosting Late Delivery Classifier ───────────────────────

# def train_late_delivery_classifier(mat_df: pd.DataFrame, mat_id: str) -> dict:
#     print(f"    Training GBM late-delivery classifier for {mat_id}...", end=" ")

#     # Binary label: 1 = late (actual LT > nominal + 1σ)
#     threshold  = mat_df["nominal_lead_time_days"].iloc[0] + mat_df["lt_deviation_days"].std()
#     y          = (mat_df["actual_lead_time_days"] > threshold).astype(int)
#     X          = mat_df[RISK_FEATURES].values

#     # Need enough positives to train
#     if y.sum() < 50:
#         print(f"insufficient late events ({y.sum()}) — skipping")
#         return None

#     X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     gbm = GradientBoostingClassifier(
#         n_estimators=200,
#         max_depth=5,
#         learning_rate=0.05,
#         subsample=0.8,
#         min_samples_leaf=20,
#         random_state=42,
#     )
#     gbm.fit(X_tr, y_tr)

#     auc  = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])
#     late_rate = y.mean()
#     print(f"done  |  AUC={auc:.3f}  |  historical late rate={late_rate:.1%}")

#     # Feature importance
#     importance = dict(zip(RISK_FEATURES, gbm.feature_importances_))
#     top_feats  = sorted(importance.items(), key=lambda x: -x[1])[:4]

#     safe_id = mat_id.replace("-", "_")
#     joblib.dump(gbm, MODELS_DIR / f"gbm_late_{safe_id}.pkl")

#     return {"model": gbm, "auc": auc, "late_rate": late_rate, "top_features": top_feats}


# def predict_late_probability(mat_id: str, feature_row: dict, gbm=None) -> dict:
#     if gbm is None:
#         safe_id = mat_id.replace("-", "_")
#         gbm     = joblib.load(MODELS_DIR / f"gbm_late_{safe_id}.pkl")

#     X   = np.array([[feature_row[f] for f in RISK_FEATURES]])
#     p_late = float(gbm.predict_proba(X)[0, 1])

#     if p_late >= RISK_THRESHOLDS["RED"]:
#         tier = "🔴  RED — HIGH probability of late delivery"
#     elif p_late >= RISK_THRESHOLDS["AMBER"]:
#         tier = "🟡  AMBER — Elevated late-delivery risk"
#     else:
#         tier = "🟢  GREEN — Delivery likely to be on time"

#     return {"p_late": round(p_late, 3), "risk_tier": tier}


# # ── Composite Risk Report ─────────────────────────────────────────────────────

# def composite_risk_report(mat_id: str, mat_df: pd.DataFrame, iso, iso_scaler, gbm_result) -> dict:
#     """Combine anomaly score + GBM late probability into a single risk summary."""
#     latest_row = mat_df.iloc[-1]

#     # Anomaly score for most recent delivery
#     X_anon    = mat_df[ANOMALY_FEATURES].values[-30:]   # last 30 days
#     scores, _ = detect_anomalies(mat_id, X_anon, iso, iso_scaler)
#     recent_anomaly_score = float(scores.mean())   # negative = more anomalous

#     # GBM late probability
#     if gbm_result and gbm_result.get("model"):
#         feature_row = latest_row[RISK_FEATURES].to_dict()
#         late_pred   = predict_late_probability(mat_id, feature_row, gbm_result["model"])
#     else:
#         late_pred = {"p_late": None, "risk_tier": "⚪  GBM not available"}

#     # Normalise anomaly score to 0–100 (higher = riskier)
#     # Typical range: -0.5 (very anomalous) to +0.5 (very normal)
#     anomaly_risk = max(0, min(100, (0.3 - recent_anomaly_score) * 100))

#     return {
#         "material_id":         mat_id,
#         "anomaly_risk_score":  round(anomaly_risk, 1),
#         "recent_anomaly_mean": round(recent_anomaly_score, 4),
#         "p_late_next_delivery":late_pred.get("p_late"),
#         "gbm_risk_tier":       late_pred["risk_tier"],
#     }


# # ── Main ──────────────────────────────────────────────────────────────────────

# def run_supplier_risk_ml(data_path="data/training_data.csv", retrain=True):
#     print("=" * 65)
#     print("  MODULE 4: SUPPLIER RISK — Isolation Forest + GradientBoosting")
#     print("=" * 65)

#     df = pd.read_csv(data_path, parse_dates=["date"])

#     all_reports = {}

#     for mat_id in df["material_id"].unique():
#         mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
#         print(f"\n  {mat_id}  ({len(mat_df):,} rows)")

#         if retrain:
#             iso, iso_scaler = train_isolation_forest(mat_df, mat_id)
#             gbm_result      = train_late_delivery_classifier(mat_df, mat_id)
#         else:
#             safe_id     = mat_id.replace("-", "_")
#             iso         = joblib.load(MODELS_DIR / f"iso_forest_{safe_id}.pkl")
#             iso_scaler  = joblib.load(MODELS_DIR / f"iso_scaler_{safe_id}.pkl")
#             gbm_result  = {"model": joblib.load(MODELS_DIR / f"gbm_late_{safe_id}.pkl")}

#         report = composite_risk_report(mat_id, mat_df, iso, iso_scaler, gbm_result)
#         all_reports[mat_id] = report

#         # Feature importance from GBM
#         if gbm_result and gbm_result.get("top_features"):
#             print(f"    GBM top risk drivers: " +
#                   "  |  ".join(f"{k}={v:.3f}" for k, v in gbm_result["top_features"]))

#     # ── Print results ──
#     print("\n" + "=" * 65)
#     print("  SUPPLIER RISK SUMMARY")
#     print("=" * 65)

#     sorted_reports = sorted(all_reports.items(), key=lambda x: -(x[1]["anomaly_risk_score"] or 0))

#     for mat_id, rep in sorted_reports:
#         p_late_str = f"{rep['p_late_next_delivery']:.0%}" if rep["p_late_next_delivery"] is not None else "N/A"
#         print(f"\n  {mat_id}")
#         print(f"  Anomaly risk score (Isolation Forest): {rep['anomaly_risk_score']}/100")
#         print(f"  P(late next delivery) [GBM]:           {p_late_str}")
#         print(f"  Risk tier:                             {rep['gbm_risk_tier']}")

#     return all_reports


# if __name__ == "__main__":
#     run_supplier_risk_ml(retrain=True)
"""
module_4_global_supplier_risk.py
ONE IsolationForest + ONE GBM across ALL materials.
material_id encoded as integer feature — shared weights, per-material bias.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path

MODELS_DIR      = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
ISO_MODEL_PATH  = MODELS_DIR / "global_iso_forest.pkl"
ISO_SCALER_PATH = MODELS_DIR / "global_iso_scaler.pkl"
GBM_MODEL_PATH  = MODELS_DIR / "global_gbm_late.pkl"
MAT_ENC_PATH    = MODELS_DIR / "global_risk_mat_encoder.pkl"
RISK_TH         = {"RED": 0.55, "AMBER": 0.30}

ANOMALY_FEATS = [
    "material_id_enc","actual_lead_time_days","lt_deviation_days",
    "transit_delay_days","po_slippage_days","qi_rejection","stock_cover_days",
]
GBM_FEATS = [
    "material_id_enc","nominal_lead_time_days","transit_risk_level",
    "po_slippage_days","qi_rejection","month","quarter","month_sin","month_cos",
    "demand_rolling_slope","daily_demand","stock_cover_days","lt_deviation_days",
]


def _encode(df, mat_enc=None, fit=False):
    df = df.copy()
    if fit:
        mat_enc = LabelEncoder()
        df["material_id_enc"] = mat_enc.fit_transform(df["material_id"])
    else:
        df["material_id_enc"] = mat_enc.transform(df["material_id"])
    return df, mat_enc


def train_global_iso(df):
    print("\n  Training GLOBAL Isolation Forest  (all materials)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[ANOMALY_FEATS].values)
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X)
    preds = iso.predict(X); n_anom = (preds == -1).sum()
    print(f"  Rows: {len(X):,}  |  Anomalies: {n_anom} ({n_anom/len(X):.1%})")
    tmp = df.copy(); tmp["_p"] = preds
    for m in sorted(df["material_id"].unique()):
        sub = tmp[tmp["material_id"]==m]; na = (sub["_p"]==-1).sum()
        print(f"    {m:<14}  {na}/{len(sub)}  ({na/len(sub):.1%})")
    joblib.dump(iso, ISO_MODEL_PATH); joblib.dump(scaler, ISO_SCALER_PATH)
    return iso, scaler


def train_global_gbm(df):
    print("\n  Training GLOBAL GBM late-delivery classifier  (all materials)...")
    df = df.copy()
    flags = []
    for m, mdf in df.groupby("material_id"):
        th = mdf["nominal_lead_time_days"].iloc[0] + mdf["lt_deviation_days"].std()
        flags.append((mdf["actual_lead_time_days"] > th).astype(int))
    df["late"] = pd.concat(flags).sort_index()
    X  = df[GBM_FEATS].values; y = df["late"].values; ix = np.arange(len(df))
    print(f"  Rows: {len(X):,}  |  Late rate: {y.mean():.1%}")
    X_tr,X_te,y_tr,y_te,ix_tr,ix_te = train_test_split(X,y,ix,test_size=0.2,random_state=42,stratify=y)
    gbm = GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                                      subsample=0.8,min_samples_leaf=20,random_state=42)
    gbm.fit(X_tr,y_tr)
    yp = gbm.predict_proba(X_te)[:,1]; auc = roc_auc_score(y_te,yp)
    print(f"  Global AUC: {auc:.3f}")
    mat_te = df["material_id"].iloc[ix_te].values
    for m in sorted(df["material_id"].unique()):
        mask = mat_te==m
        if mask.sum()>0 and y_te[mask].sum()>0:
            print(f"    {m:<14}  AUC={roc_auc_score(y_te[mask],yp[mask]):.3f}  "
                  f"late={y_te[mask].mean():.1%}")
    top5 = sorted(zip(GBM_FEATS,gbm.feature_importances_),key=lambda x:-x[1])[:5]
    print("  Top features: "+"  |  ".join(f"{k}={v:.3f}" for k,v in top5))
    joblib.dump(gbm, GBM_MODEL_PATH)
    return {"model": gbm, "auc": auc}


def score_risk(mat_id, recent_df, iso=None, iso_sc=None, gbm=None, mat_enc=None):
    if iso is None:
        iso=joblib.load(ISO_MODEL_PATH); iso_sc=joblib.load(ISO_SCALER_PATH)
        gbm=joblib.load(GBM_MODEL_PATH); mat_enc=joblib.load(MAT_ENC_PATH)
    d = recent_df.copy(); d["material_id_enc"] = int(mat_enc.transform([mat_id])[0])
    X_iso = iso_sc.transform(d[ANOMALY_FEATS].values[-30:])
    anom  = float(max(0,min(100,(0.3-iso.decision_function(X_iso).mean())*100)))
    X_gbm = np.array([[d.iloc[-1][f] for f in GBM_FEATS]])
    p_late= float(gbm.predict_proba(X_gbm)[0,1])
    tier  = ("🔴  RED — HIGH probability of late delivery" if p_late>=RISK_TH["RED"] else
             "🟡  AMBER — Elevated late-delivery risk"     if p_late>=RISK_TH["AMBER"] else
             "🟢  GREEN — Delivery likely on time")
    return {"material_id":mat_id,"anomaly_risk_score":round(anom,1),
            "p_late_next_delivery":round(p_late,3),"risk_tier":tier}


def run_global_supplier_risk(data_path="data/training_data.csv", retrain=True):
    print("="*65+"\n  MODULE 4 (Global): SUPPLIER RISK — IsoForest + GBM\n"+"="*65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    df, mat_enc = _encode(df, fit=True)
    if retrain or not ISO_MODEL_PATH.exists():
        iso, iso_sc = train_global_iso(df)
        gbm = train_global_gbm(df)["model"]
        joblib.dump(mat_enc, MAT_ENC_PATH)
    else:
        print("  Loading saved global models...")
        iso=joblib.load(ISO_MODEL_PATH); iso_sc=joblib.load(ISO_SCALER_PATH)
        gbm=joblib.load(GBM_MODEL_PATH); mat_enc=joblib.load(MAT_ENC_PATH)
    reports = {}
    print("\n"+"="*65+"\n  SUPPLIER RISK REPORT  (one global model)\n"+"="*65)
    for m in sorted(df["material_id"].unique()):
        rep = score_risk(m, df[df["material_id"]==m].sort_values("date"),
                         iso,iso_sc,gbm,mat_enc)
        reports[m] = rep
        print(f"\n  {m}\n  Anomaly risk: {rep['anomaly_risk_score']}/100  "
              f"P(late): {rep['p_late_next_delivery']:.0%}\n  {rep['risk_tier']}")
    return reports


if __name__ == "__main__":
    run_global_supplier_risk(retrain="--no-retrain" not in sys.argv)