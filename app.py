import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Real-time Stock Prediction BBCA (LSTM & TCN)")

# SETTING LOOKBACK TETAP (Sesuai settingan training skripsi kamu)
LOOKBACK_FIXED = 60

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    model_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    model_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    model_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return model_harian, model_mingguan, model_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi
def predict_future(model, full_data, lookback=LOOKBACK_FIXED):
    scaler = RobustScaler()
    scaled_all = scaler.fit_transform(full_data.reshape(-1, 1))
    
    if len(scaled_all) < lookback:
        return None
        
    last_sequence = scaled_all[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Sidebar (Hanya Informasi)
with st.sidebar:
    st.write("### Informasi Aplikasi")
    st.info(f"Aplikasi ini menggunakan jendela waktu (lookback) sebanyak {LOOKBACK_FIXED} hari untuk memprediksi harga selanjutnya.")

# 5. Main UI & Data Scraping
st.subheader("Analisis Harga Real-time (BBCA.JK)")
# Ambil data 1 tahun agar data mencukupi tanpa error "Data tidak cukup"
df = yf.download("BBCA.JK", period='1y', interval='1d') 

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0]
    else:
        close_data = df['Close']

    # Visualisasi Grafik
    st.area_chart(close_data.tail(100))
    
    # Menampilkan Harga Terakhir
    last_price = float(close_data.iloc[-1])
    last_date = df.index[-1].date()
    
    st.metric(label="Harga Terakhir (Market Close)", 
              value=f"Rp {last_price:,.2f}", 
              delta=f"Tanggal: {last_date}")

    with st.expander("Lihat Tabel Data Historis"):
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

    # Tombol Prediksi Langsung
    if st.button('Mulai Prediksi Harga Selanjutnya'):
        with st.spinner('Model sedang menganalisis pola...'):
            latest_prices = close_data.values
            hasil = predict_future(model_h, latest_prices)
            
            if hasil:
                st.success(f"### Hasil Prediksi Hari Selanjutnya: Rp {hasil:,.2f}")
                st.balloons() # Efek perayaan kecil
            else:
                st.error("Gagal melakukan prediksi. Data historis tidak mencukupi.")
else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
