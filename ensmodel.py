import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ---------------- Encryption Function ---------------- #
def encrypt_text(text: str, key: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(text.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()

# ---------------- Load Dataset ---------------- #
df = pd.read_csv('ddos_dataset.csv')  # Replace with your dataset path

# ---------------- Encrypt patient_info column (if exists) ---------------- #
if 'patient_info' in df.columns:
    aes_key = os.urandom(16)
    df['encrypted_info'] = df['patient_info'].apply(lambda x: encrypt_text(str(x), aes_key))

# ---------------- Advanced Feature Engineering ---------------- #
for col in df.select_dtypes(include='object').columns:
    if col != 'label':
        df[col] = LabelEncoder().fit_transform(df[col])

df['label'] = LabelEncoder().fit_transform(df['label'])

# Create new interaction features
if 'pkt_size' in df.columns and 'duration' in df.columns:
    df['pkt_density'] = df['pkt_size'] / (df['duration'] + 1e-5)

if 'flag_syn' in df.columns and 'protocol' in df.columns:
    df['syn_protocol'] = df['flag_syn'] * df['protocol']

# ---------------- Train/Test Split ---------------- #
X = df.drop(['label'], axis=1)
y = df['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------- Ensemble Model ---------------- #
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb = LGBMClassifier(random_state=42)


ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb),
    ('lgb', lgb),

], voting='soft')

# ---------------- Train and Evaluate ---------------- #
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- Save Model & Scaler ---------------- #
joblib.dump(ensemble, 'ddos_ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ---------------- Visualization ---------------- #
labels = ['Normal', 'Attack']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - DDoS Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Bar chart for predictions
_, counts = np.unique(y_pred, return_counts=True)
plt.bar(labels, counts, color=['green', 'red'])
plt.title("Prediction Distribution (Normal vs Attack)")
plt.ylabel("Count")
plt.show()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 🔹 Bar chart: count of predictions
unique, counts = pd.Series(y_pred).value_counts().sort_index().items()
labels = ['Normal', 'Attack']

plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=['green', 'red'])
plt.title("Prediction Distribution: Attack vs Normal")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
