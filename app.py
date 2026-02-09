import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# Pastikan library autorefresh terpasang di requirements.txt
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_error = "Library 'streamlit-autorefresh' belum terinstal."

# 1. Konfigurasi Halaman & Auto-refresh (Setiap 1 Detik)
st.set_page_config(page_title="Dashboard Skripsi BBCA", layout="wide")
st_autorefresh(interval=1000, key="datarefresh")

# --- LOGIKA WAKTU REAL-TIME (WIB) ---
tz_jkt = pytz.timezone('Asia/Jakarta')
now_jkt = datetime.now(tz_jkt)

st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")
st.write(f"**Waktu Sistem (Real-time):** `{now_jkt.strftime('%H:%M:%S')}` WIB | **Tanggal:** `{now_jkt.strftime('%d-%m-%Y')}`")

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    if len(scaled_data) < lookback: return None
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence)
    return scaler.inverse_transform(prediction_scaled)[0][0]

# 4. Penarikan Data (5 tahun)
df_all = yf.download("BBCA.JK", period='5y')

if not df_all.empty:
    # Ambil Close saja untuk mesin prediksi
    close_series = df_all['Close'].iloc[:, 0] if isinstance(df_all.columns, pd.MultiIndex) else df_all['Close']
    close_series = close_series.dropna()

    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN (LSTM) ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian (LSTM)")
        last_p = float(close_series.iloc[-1])
        # Prediksi untuk hari ini (backtest)
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, lookback=60)

        c1, c2 = st.columns(2)
        with c1: st.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}"); st.caption("Status: Real-time WIB")
        with c2: st.metric("Prediksi LSTM", f"Rp {pred_today:,.2f}")
        
        # TOMBOL PREDIKSI BESOK (DIKEMBALIKAN)
        if st.button('🔮 Jalankan Prediksi LSTM (Besok)'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_h, close_series.values, lookback=60)
                st.success(f"### Estimasi Harga LSTM Besok: Rp {hasil:,.2f}")
        
        with st.expander("Lihat Data Historis Harian Lengkap (OHLCV)"):
            st.dataframe(df_all.sort_index(ascending=False), use_container_width=True)

    # --- TAB 2: MINGGUAN (TCN) ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan (TCN)")
        df_w_full = df_all.resample('W-MON').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_w = float(df_w_full['Close'].iloc[-1])
        pred_w = predict_stock(model_m, df_w_full['Close'].values[:-1], lookback=24)

        col1, col2 = st.columns(2)
        with col1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
        with col2: st.metric("Prediksi TCN (Minggu Ini)", f"Rp {pred_w:,.2f}")

        # TOMBOL PREDIKSI MINGGU DEPAN (DIKEMBALIKAN)
        if st.button('🔮 Jalankan Prediksi TCN (Minggu Depan)'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_m, df_w_full['Close'].values, lookback=24)
                st.success(f"### Estimasi Harga TCN Minggu Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Mingguan Lengkap (OHLCV)"):
            st.dataframe(df_w_full.sort_index(ascending=False), use_container_width=True)

    # --- TAB 3: BULANAN (TCN) ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan (TCN)")
        df_m_full = df_all.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_m = float(df_m_full['Close'].iloc[-1])
        pred_m = predict_stock(model_b, df_m_full['Close'].values[:-1], lookback=12)

        k1, k2 = st.columns(2)
        with k1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
        with k2: st.metric("Prediksi TCN (Bulan Ini)", f"Rp {pred_m:,.2f}")

        # TOMBOL PREDIKSI BULAN DEPAN (DIKEMBALIKAN)
        if st.button('🔮 Jalankan Prediksi TCN (Bulan Depan)'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_b, df_m_full['Close'].values, lookback=12)
                st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Bulanan Lengkap (OHLCV)"):
            st.dataframe(df_m_full.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Gagal mengambil data.")
