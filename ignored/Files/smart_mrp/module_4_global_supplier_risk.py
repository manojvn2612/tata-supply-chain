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
