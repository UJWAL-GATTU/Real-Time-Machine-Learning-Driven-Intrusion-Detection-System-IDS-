import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score , roc_curve
import joblib
import os

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
    stratify: bool = True , # This means as we are dealing with an imbalanced dataset we want to maintain the same proportion of classes in both train and test sets as in the original dataset
    ):

    '''stratify means to maintain the same proportion of classes in both train and test sets as in the original dataset which is important for imbalanced datasets
    This is essential for comparing different models or hyperparameter tuning experiments, as it eliminates variations in results due to different data splits.''' 

    '''
    Convenience wrapper around train_test_split.
    By default, it stratifies on y to keep normal/attack ratio similar.
    '''
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
    return X_train, X_test, y_train, y_test


csv_path = r"C:\\Users\\Gattu Ujwal\\Desktop\\ML_Journey\\Anamoly_Detection_In_Network_Traffic\\processed_data.csv"  # TODO: change this
df = pd.read_csv(csv_path)

# ===== 1. Build X and y =====
# Features and target
X = df.drop(columns=[LABEL_COL]).values
y = df[LABEL_COL].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of features: {X.shape[1]}")

# ===== 2. Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class ratio same in train and test
)

print(f"Train: {X_train.shape} Test: {X_test.shape}")

# ===== 3. Scaling =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled shapes: {X_train_scaled.shape} {X_test_scaled.shape}")

# ===== 4. PCA (optional but you were using it) =====
# Keep 19 components like your previous experiment
pca = PCA(n_components=19, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA shapes: {X_train_pca.shape} {X_test_pca.shape}")
print(f"Number of PCA components: {pca.n_components_}")

# ===== 5. Random Forest (supervised) =====
rf = RandomForestClassifier(
    n_estimators=300,       # number of trees
    class_weight='balanced',  # handle class imbalance
    n_jobs=-1,             # use all CPU cores
    random_state=42
)

print("Training RandomForestClassifier...")
rf.fit(X_train_pca, y_train)

# ===== 6. Evaluation =====
y_pred = rf.predict(X_test_pca)
y_proba = rf.predict_proba(X_test_pca)[:, 1]  # prob for class 1

print("\nClassification report (RandomForest):")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)


# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# === Precision-Recall Curve ===
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

plt.figure()
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.grid(True)
plt.legend()
plt.show()

# === Confusion Matrix Heatmap ===
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ===== 7. Feature Importance =====
feature_cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
importances = rf.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:15]:
    print(f"{name}: {imp:.4f}")

#=== 8. Save models =====
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca.joblib")
joblib.dump(rf, "models/random_forest.joblib")
print("Models saved to 'models/' directory.")