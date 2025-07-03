import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model
model = joblib.load("ddos_ensemble_model.pkl")

st.title("Healthcare DDoS Detection Interface")
st.write("Upload a network traffic CSV (e.g., from firewall or router logs)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw data preview:")
    st.dataframe(df.head())

    # Preprocess
    for col in df.select_dtypes(include='object').columns:
        if col != 'label':
            df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop('label', axis=1, errors='ignore')

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    df['Prediction'] = predictions
    st.success("Prediction complete!")
    st.write(df[['Prediction']].value_counts())

    st.download_button("Download Results", df.to_csv(index=False), "predicted_ddos.csv", "text/csv")
