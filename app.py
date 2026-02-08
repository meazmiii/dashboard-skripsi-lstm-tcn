import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Real-time Stock Prediction BBCA (LSTM & TCN)")

@st.cache_resource
def load_models():
    # Pastikan nama file di sini sama persis dengan yang ada di folder/gambar
    model_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    model_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    model_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return model_harian, model_mingguan, model_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

def predict_future(model, full_data, lookback=60):
    scaler = RobustScaler()
    # Fit pada seluruh data yang tersedia agar range harga lebih akurat
    scaled_all = scaler.fit_transform(full_data.reshape(-1, 1))
    
    # Ambil jendela terakhir
    last_sequence = scaled_all[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

st.subheader("Data Saham Terkini (BBCA.JK)")
# Ambil data lebih panjang (misal 5thn) agar Scaler lebih stabil
df = yf.download("BBCA.JK", period='5y')

if not df.empty:
    # Mengatasi MultiIndex kolom pada yfinance terbaru
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Adj Close']['BBCA.JK']
    else:
        close_data = df['Adj Close']

    st.line_chart(close_data.tail(100))

    if st.button('Mulai Prediksi Harga Harian'):
        with st.spinner('Menganalisis pola...'):
            latest_prices = close_data.values
            hasil = predict_future(model_h, latest_prices, lookback=lookback_val)
            
            st.success(f"Hasil Prediksi Harga Selanjutnya: Rp {hasil:,.2f}")
            st.info(f"Tanggal Data Terakhir: {df.index[-1].date()}")
