# Real-Time-Machine-Learning-Driven-Intrusion-Detection-System-IDS-

A Real-Time Network Intrusion Detection System that uses Machine Learning and flow-based features to detect malicious network traffic.
The system combines offline training using the UNSW-NB15 dataset and real-time packet capture to classify suspicious flows and generate alerts.

📌 Project Overview

This project implements a behavior-based IDS using:

Flow-level feature engineering

Supervised Machine Learning (Random Forest)

Unsupervised anomaly detection (Isolation Forest)

Real-time network packet sniffing using Scapy

Attack probability scoring and alert logging

The IDS predicts whether a network flow is normal or malicious and stores suspicious flows in separate log files.

# Key Features

Real-time packet capturing and flow creation

41-feature ML-based intrusion detection model

Random Forest classifier (high accuracy)

PCA for dimensionality reduction

Attack probability scoring

Malicious flow logging:

detected_attacks.csv

alerts.log

Works on live LAN traffic

# System Architecture
UNSW-NB15 Dataset ──► Preprocessing & Feature Engineering ─► PCA ─► Train ML Model ─► Save Model

Live Packet Sniffing ─► Flow Extraction ─► Feature Vector ─► Scaling + PCA ─► ML Prediction ─► Alerts

# Dataset Used – UNSW-NB15

A modern network intrusion dataset containing:

Normal traffic + multiple attack categories

Realistic flow-based statistics

49 raw features (33 selected)

Used as the ground truth for training ML models

# Feature Engineering

Raw Features:
Metrics such as duration, packet counts, byte counts, load, TTL, flags, etc.

Engineered Features:

bytes_total

pkts_total

avg_pkt_size

pps (packets per second)

bps (bits per second)

pkt_ratio

byte_ratio

Total Features Used: 41

# Machine Learning Models
Random Forest (Supervised)

Primary model

High accuracy and ROC-AUC

Excellent at classifying malicious flows

Isolation Forest (Unsupervised)

Detects anomalies without labels

Useful for unknown or zero-day behavior

Saved Components
scaler.joblib
pca.joblib
random_forest.joblib

# Real-Time Detection

Packets captured using Scapy

Flows built using IP, port, protocol, timing

Features generated to match training schema

Prediction performed using loaded ML models

Alerts generated for any detected malicious flow

Example alert:

[ALERT] src=192.168.1.10 dst=8.8.8.8 prob=0.9453

# How to Run
Install dependencies:
pip install scikit-learn pandas numpy joblib scapy

Run live detection:
python live_sniffer.py

Run predictions on CSV:
python new_predictions.py

# Model Performance (Offline)

Accuracy: 93%

Precision (attack): 96%

Recall (attack): 91%

ROC-AUC: 0.985

# Output Files

detected_attacks.csv → Stores all suspicious flows

alerts.log → Logs attack alerts with probability and IP details

# Challenges

Flow extraction limitations on Windows

Matching live features with training schema

Dataset imbalance impacting unsupervised learning

Future Improvements

Web dashboard for monitoring

Deep learning models (LSTM, Autoencoder)

SIEM integration

High-speed streaming pipeline (Kafka)

Payload analysis for deeper inspection
