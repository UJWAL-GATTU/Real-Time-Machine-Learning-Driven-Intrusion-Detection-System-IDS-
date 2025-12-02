import numpy as np 
import pandas as pd
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix


# Paths - CHANGE IF NEEDED

MODEL_DIR = r"C:\Users\Gattu Ujwal\Desktop\ML_Journey\Anamoly_Detection_In_Network_Traffic\models"
DATA_PATH = r"C:\Users\Gattu Ujwal\Downloads\live_capture.csv"

# Load models

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
pca    = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))
rf     = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))

print("Loaded scaler, PCA, and RandomForest model.")

# Training feature columns

raw_training_features = [
    "dur","spkts","dpkts","sbytes","dbytes","smean","dmean",
    "sload","dload","sloss","dloss","rate","tcprtt","synack","ackdat",
    "sttl","dttl","sinpkt","dinpkt","sjit","djit",
    "swin","dwin","stcpb","dtcpb",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
    "ct_dst_sport_ltm","ct_dst_src_ltm","ct_src_ltm","ct_srv_dst",
    "proto"
]

engineered_cols = [
    "bytes_total","pkts_total","avg_pkt_size",
    "pps","bps","pkt_ratio","byte_ratio"
]

training_feature_cols = raw_training_features + engineered_cols  # 41 features

# Load new data

df = pd.read_csv(DATA_PATH)
print("Loaded data:", df.shape)

has_label = "label" in df.columns
if has_label:
    y_true = df["label"].values
    df_features = df.drop(columns=["label"]).copy()
else:
    y_true = None
    df_features = df.copy()

# Add missing raw features (fill with 0)

for col in raw_training_features:
    if col not in df_features.columns:
        df_features[col] = 0.0

for col in ["dur", "spkts", "dpkts", "sbytes", "dbytes"]:
    if col in df_features.columns:
        df_features[col] = df_features[col].astype(float)

# Recompute engineered features

df_features["bytes_total"] = df_features["sbytes"] + df_features["dbytes"]
df_features["pkts_total"]  = df_features["spkts"] + df_features["dpkts"]

df_features["avg_pkt_size"] = df_features["bytes_total"] / df_features["pkts_total"].replace(0, 1.0)
df_features["avg_pkt_size"] = df_features["avg_pkt_size"].fillna(0.0)

safe_dur = df_features["dur"].replace(0, 1e-3)
df_features["pps"] = df_features["pkts_total"] / safe_dur
df_features["bps"] = (8.0 * df_features["bytes_total"]) / safe_dur

df_features["pkt_ratio"] = df_features["spkts"] / df_features["dpkts"].replace(0, 1.0)
df_features["byte_ratio"] = df_features["sbytes"] / df_features["dbytes"].replace(0, 1.0)

# Reorder columns

X_raw = df_features[training_feature_cols].values
print("Final feature matrix shape:", X_raw.shape)

# Scale → PCA → Predict

X_scaled = scaler.transform(X_raw)
X_pca    = pca.transform(X_scaled)

y_pred = rf.predict(X_pca)
y_proba = rf.predict_proba(X_pca)[:, 1]

print("Prediction completed.")

attack_mask = (y_pred == 1)
attack_count = int(attack_mask.sum())
normal_count = int((y_pred == 0).sum())

print("Predicted attack count:", attack_count)
print("Predicted normal count:", normal_count)


attack_df = df[df["pred_label"] == 1]

# Save detected attacks to CSV (only if at least one attack exists)
if len(attack_df) > 0:
    attack_df.to_csv("detected_attacks.csv", index=False)

    with open("alerts.log", "a") as f:
        for idx, row in attack_df.iterrows():
            src = row["src_ip"] if "src_ip" in row else "N/A"
            dst = row["dst_ip"] if "dst_ip" in row else "N/A"
            prob = row["attack_prob"]
            f.write(f"[ALERT] src={src} dst={dst} prob={prob:.4f}\n")

    print(f"\n⚠ ALERTS logged: {len(attack_df)} attacks found.")
    print("Saved detailed attacks to: detected_attacks.csv")
    print("Appended alert messages to: alerts.log\n")

else:
    print("\n✅ No attacks detected. No alert file generated.\n")

# ============================
# Save full prediction results
# ============================
OUT_PATH = os.path.join(os.path.dirname(DATA_PATH), "synthetic_predictions.csv")
df.to_csv(OUT_PATH, index=False)

print("Saved full predictions to:", OUT_PATH)


if has_label:
    print("\nEvaluation on synthetic data:")
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

# Save results

df["pred_label"] = y_pred
df["attack_prob"] = y_proba

OUT_PATH = os.path.join(os.path.dirname(DATA_PATH), "synthetic_predictions.csv")
df.to_csv(OUT_PATH, index=False)

print("Saved predictions to:", OUT_PATH)
