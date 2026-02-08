import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Real-time Stock Prediction BBCA (LSTM & TCN)")

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    # Nama file disesuaikan persis dengan folder skripsi kamu
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi Universal
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    if len(scaled_data) < lookback:
        return None
        
    last_sequence = scaled_data[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    pred_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(pred_scaled)
    return prediction[0][0]

# 4. Sidebar Informasi
with st.sidebar:
    st.write("### Informasi Dashboard")
    st.info("Dashboard ini memprediksi harga saham BBCA menggunakan 3 model berbeda sesuai timeframe skripsi.")

# 5. Penarikan Data (Ambil 2 tahun agar data mingguan/bulanan cukup)
df_raw = yf.download("BBCA.JK", period='2y')

if not df_raw.empty:
    # Ambil kolom Close secara aman
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_series = df_raw['Close'].iloc[:, 0]
    else:
        close_series = df_raw['Close']

    # --- BAGIAN TABS UNTUK 3 TIMEFRAME ---
    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Prediksi Harga Harian")
        st.line_chart(close_series.tail(90)) # Menampilkan 90 hari terakhir
        
        last_p = close_series.iloc[-1]
        st.metric("Harga Terakhir", f"Rp {last_p:,.2f}", f"Update: {df_raw.index[-1].date()}")
        
        if st.button('Prediksi Hari Besok'):
            hasil = predict_stock(model_h, close_series.values, lookback=60)
            if hasil:
                st.success(f"Prediksi Harga Selanjutnya: Rp {hasil:,.2f}")
                st.balloons()

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Prediksi Harga Mingguan")
        # Resample ke mingguan (W-MON = Weekly ending Monday)
        df_weekly = close_series.resample('W-MON').last()
        st.line_chart(df_weekly.tail(52)) # Menampilkan 1 tahun (52 minggu)
        
        if st.button('Prediksi Minggu Depan'):
            # Gunakan lookback sesuai training mingguan (misal: 30)
            hasil = predict_stock(model_m, df_weekly.values, lookback=30)
            if hasil:
                st.success(f"Prediksi Harga Minggu Depan: Rp {hasil:,.2f}")
            else:
                st.error("Data mingguan tidak cukup (Butuh minimal 30 minggu).")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Prediksi Harga Bulanan")
        # Resample ke bulanan (M = Month end)
        df_monthly = close_series.resample('M').last()
        st.line_chart(df_monthly.tail(24)) # Menampilkan 2 tahun (24 bulan)
        
        if st.button('Prediksi Bulan Depan'):
            # Gunakan lookback sesuai training bulanan (misal: 12)
            hasil = predict_stock(model_b, df_monthly.values, lookback=12)
            if hasil:
                st.success(f"Prediksi Harga Bulan Depan: Rp {hasil:,.2f}")
