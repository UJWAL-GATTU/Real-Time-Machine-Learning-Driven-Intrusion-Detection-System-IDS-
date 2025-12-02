# 🚨 Real-Time Machine Learning–Driven Intrusion Detection System (IDS)

A real-time Intrusion Detection System that uses Machine Learning and flow-based feature engineering to detect malicious network traffic.  
The system integrates offline ML training using the UNSW-NB15 dataset and real-time packet sniffing using Scapy.

---

## 📌 Project Overview

- Flow-based intrusion detection  
- 41-feature ML classification model  
- Random Forest (supervised) + Isolation Forest (unsupervised)  
- Real-time packet sniffing using Scapy  
- Attack probability scoring  
- Logging malicious flows with IP addresses  

---

## ✨ Features

- Real-time network traffic analysis  
- Flow creation using (src_ip, dst_ip, ports, protocol)  
- Feature engineering (41 features)  
- PCA for dimensionality reduction  
- Random Forest for high-accuracy classification  
- Attack logging:
  - `detected_attacks.csv`
  - `alerts.log`  
- Works on LAN environments for behavioral intrusion detection  

---

## 📊 Dataset Used — UNSW-NB15

- Modern cybersecurity dataset  
- Contains normal + 9 types of attacks  
- 49 raw features  
- Used for model training and evaluation  

---

## 🧮 Feature Engineering

**Raw Features (selected):**  
`dur, spkts, dpkts, sbytes, dbytes, smean, dmean, sttl, dttl, rate, sjit, djit, stcpb, dtcpb, proto, ...`

**Engineered Features:**  
- `bytes_total`  
- `pkts_total`  
- `avg_pkt_size`  
- `pps`  
- `bps`  
- `pkt_ratio`  
- `byte_ratio`

**Total Features:** 41

---

## 🤖 Machine Learning Models

### Random Forest (Primary Model)
- Accuracy: **93%**
- ROC-AUC: **0.985**
- High recall for detecting attacks

### Isolation Forest (Anomaly Model)
- Detects unknown or zero-day anomalies
- Provides anomaly scores

### Saved Model Files

- scaler.joblib
- pca.joblib
- random_forest.joblib


---

## 🖥 Real-Time Detection Pipeline

1. Capture packets using Scapy  
2. Convert packets → flows  
3. Generate 41 features  
4. Scale → PCA → Predict  
5. Log malicious flows  

**Example Alert:**
[ALERT] src=192.168.1.5 dst=8.8.8.8 prob=0.9471


---

## Installation

```bash
pip install scikit-learn pandas numpy joblib scapy
```
##  Train & preprocess dataset

```bash
python preprocess_and_train.py
```

## Run real-time IDS

```bash
python live_sniffer.py
```

## Predict on CSV file

```bash
python new_predictions.py
```

## 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 93%   |
| Precision | 96%   |
| Recall    | 91%   |
| ROC-AUC   | 0.985 |


🚨 Output Files

detected_attacks.csv   → All detected malicious flows
alerts.log             → Human-readable alert messages


🚀 Future Enhancements

Web dashboard for real-time alerts
Deep learning (Autoencoder / LSTM)
Kafka-based high-speed streaming
Payload analysis and NLP-based IDS
Integration with SIEM platforms
