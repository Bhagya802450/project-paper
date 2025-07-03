import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import joblib

# 🔹 Load dataset
df = pd.read_csv("ddos_dataset.csv")

# 🔹 Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    if col != 'label':
        df[col] = LabelEncoder().fit_transform(df[col])

# 🔹 Encode label
df['label'] = LabelEncoder().fit_transform(df['label'])

# 🔹 Split features/target
X = df.drop('label', axis=1)
y = df['label']

# 🔹 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Save the scaler
joblib.dump(scaler, "advanced_scaler.pkl")

# 🔹 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Define models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb = LGBMClassifier(random_state=42)

# 🔹 Ensemble model
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb),
    ('lgb', lgb)
], voting='soft')

# 🔹 Train and evaluate
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))

# 🔹 Save the model
joblib.dump(ensemble, "advanced_ddos_ensemble_model.pkl")
