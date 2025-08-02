
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="EV Charging Demand Forecast", layout="centered")
st.title("🔋 EV Charging Demand Forecasting")

model = pickle.load(open("ev_model.pkl", "rb"))

uploaded_file = st.file_uploader("Upload your EV data file (CSV format)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.write(data.head())

    for col in ['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)']:
        data[col] = data[col].astype(str).str.replace(',', '', regex=True).astype(float)

    X = data[['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)']]
    predictions = model.predict(X)
    data['Predicted EV Demand'] = predictions

    st.subheader("📊 Predicted EV Demand")
    st.write(data[['County', 'State', 'Predicted EV Demand']])

    st.download_button("Download Predictions as CSV", data.to_csv(index=False), "ev_predictions.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")
