import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler # standardscaler is used to standardize the features by making their mean 0 and variance 1
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


LABEL_COL = "label" 

def load_processed_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(r"C:\\Users\\Gattu Ujwal\\Desktop\\ML_Journey\\Anamoly_Detection_In_Network_Traffic\\processed_data.csv")
    return df

def get_feature_and_label_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Return list of feature column names (X) and label column name (y).
    Here: all columns except 'label' are treated as features.
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"'{LABEL_COL}' column not found in DataFrame")
    
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    return feature_cols, LABEL_COL


def build_X_y(df: pd.DataFrame):
    """
    Given a processed DataFrame, return:
      X : 2D numpy array of features
      y : 1D numpy array of labels
      feature_cols : list of feature column names
    """
    feature_cols, label_col = get_feature_and_label_columns(df)
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y, feature_cols


def train_test_split_Xy(
    X,
    y,
    test_size: float = 0.2, #splitting 20% of data for testing and 80% for training
    random_state: int = 42, # for reproducibility it means every time we execute the code the same random split will be generated
    stratify: bool = True ,
    ):

    '''stratify means to maintain the same proportion of classes in both train and test sets as in the original dataset which is important for imbalanced datasets
    This is essential for comparing different models or hyperparameter tuning experiments, as it eliminates variations in results due to different data splits.''' 

    """
    Convenience wrapper around train_test_split.
    By default, it stratifies on y to keep normal/attack ratio similar.
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
    return X_train, X_test, y_train, y_test


path = (r"C:\\Users\\Gattu Ujwal\\Desktop\\ML_Journey\\Anamoly_Detection_In_Network_Traffic\\processed_data.csv")

# 1) Load processed data
df = load_processed_csv(path)

# 2) Build X (features) and y (labels)
X, y, feature_cols = build_X_y(df)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of features:", len(feature_cols))

# 3) Split into train and test (for evaluation)
X_train, X_test, y_train, y_test = train_test_split_Xy(X, y)
print("Train:", X_train.shape, "Test:", X_test.shape)


# 4) Scale the features

scaler = StandardScaler()
scaler.fit(X_train)  # fit only on training data

X_train_scaled = scaler.transform(X_train) 
X_test_scaled  = scaler.transform(X_test)

print("Scaled shapes:", X_train_scaled.shape, X_test_scaled.shape)


# 5) PCA for dimensionality reduction
# Keep 95% variance (you can tweak this)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("PCA shapes:", X_train_pca.shape, X_test_pca.shape)
print("Number of PCA components:", pca.n_components_)


# 6) Train IsolationForest (on NORMAL traffic only)

# In UNSW, usually: 0 = normal, 1 = attack
normal_mask = (y_train == 0)
X_train_normal = X_train_pca[normal_mask]

print("Normal train samples:", X_train_normal.shape[0])

iso = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.3,   # expected proportion of anomalies (tune later)
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_normal)
print("IsolationForest trained.")


# 7) Get anomaly scores on test set

# decision_function: higher = more normal, lower = more anomalous
# We invert it so that HIGHER = more anomalous
test_scores_raw = iso.decision_function(X_test_pca)
anomaly_scores = -test_scores_raw  # higher = more anomalous

# Choose a threshold:
# Option A: quantile-based (e.g., top 2% as anomalies)
threshold = np.quantile(anomaly_scores, 0.98)
print("Anomaly score threshold:", threshold)

y_pred = (anomaly_scores > threshold).astype(int)  # 1 = anomaly, 0 = normal

# 8) Evaluate

print("\nClassification report (IsolationForest, threshold-based):")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

try:
    print("ROC-AUC (using raw anomaly scores):",
          roc_auc_score(y_test, anomaly_scores))
except Exception as e:
    print("ROC-AUC could not be computed:", e)

# Optional: simple visualization of anomaly score distribution
plt.figure(figsize=(8,4))
plt.hist(anomaly_scores, bins=50)
plt.axvline(threshold, color='red', linestyle='--', label='threshold')
plt.title("Anomaly score distribution (test set)")
plt.xlabel("Anomaly score (higher = more anomalous)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()
