import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from streamlit_autorefresh import st_autorefresh

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi BBCA", layout="wide")

# Jam Real-time (Refresh setiap 1 detik)
st_autorefresh(interval=1000, key="clock_refresh")
tz_jkt = pytz.timezone('Asia/Jakarta')
now_jkt = datetime.now(tz_jkt)

st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")
st.write(f"**Waktu Sistem (Real-time):** `{now_jkt.strftime('%H:%M:%S')}` WIB | **Tanggal:** `{now_jkt.strftime('%d-%m-%Y')}`")

# 2. Fungsi Load Model & Data (Gunakan Cache agar stabil)
@st.cache_resource
def load_models():
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

@st.cache_data(ttl=3600)
def get_data():
    df = yf.download("BBCA.JK", period='5y')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    model_h, model_m, model_b = load_models()
    df_all = get_data()
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

# 3. Fungsi Prediksi
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    if len(scaled_data) < lookback: return None
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence)
    return scaler.inverse_transform(prediction_scaled)[0][0]

if not df_all.empty:
    close_series = df_all['Close'].dropna()
    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN (Tetap ada Histori 5 Hari) ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian (LSTM)")
        last_p = float(close_series.iloc[-1])
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, lookback=60)

        c1, c2 = st.columns(2)
        with c1: st.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}"); st.caption("Status: Real-time WIB")
        with c2: st.metric("Prediksi LSTM", f"Rp {pred_today:,.2f}")
        
        st.write("### 🕒 Historis Akurasi (5 Hari Bursa Terakhir)")
        history_h = []
        for i in range(1, 6):
            t_idx = -i
            act_val = close_series.iloc[t_idx]
            p_val = predict_stock(model_h, close_series.iloc[:t_idx].values, lookback=60)
            history_h.append({
                "Tanggal": close_series.index[t_idx].date(),
                "Harga Aktual": f"Rp {act_val:,.2f}",
                "Prediksi LSTM": f"Rp {p_val:,.2f}",
                "Selisih (Rp)": f"{abs(act_val - p_val):,.2f}"
            })
        st.table(pd.DataFrame(history_h))

        if st.button('Jalankan Prediksi LSTM (Besok)'):
            hasil = predict_stock(model_h, close_series.values, lookback=60)
            st.success(f"### Estimasi Harga LSTM Besok: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Harian Lengkap"):
            st.dataframe(df_all.sort_index(ascending=False), use_container_width=True)

    # --- TAB 2: MINGGUAN (Tanpa Histori 5 Terakhir) ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan (TCN)")
        df_w_full = df_all.resample('W-MON').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_w = float(df_w_full['Close'].iloc[-1])
        pred_w = predict_stock(model_m, df_w_full['Close'].values[:-1], lookback=24)

        col1, col2 = st.columns(2)
        with col1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
        with col2: st.metric("Prediksi TCN", f"Rp {pred_w:,.2f}")

        st.markdown("---")
        if st.button('Jalankan Prediksi TCN (Minggu Depan)'):
            hasil = predict_stock(model_m, df_w_full['Close'].values, lookback=24)
            st.success(f"### Estimasi Harga TCN Minggu Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Mingguan Lengkap (OHLCV)"):
            st.dataframe(df_w_full.sort_index(ascending=False), use_container_width=True)

    # --- TAB 3: BULANAN (Tanpa Histori 5 Terakhir) ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan (TCN)")
        df_m_full = df_all.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_m = float(df_m_full['Close'].iloc[-1])
        pred_m = predict_stock(model_b, df_m_full['Close'].values[:-1], lookback=12)

        k1, k2 = st.columns(2)
        with k1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
        with k2: st.metric("Prediksi TCN", f"Rp {pred_m:,.2f}")

        st.markdown("---")
        if st.button('Jalankan Prediksi TCN (Bulan Depan)'):
            hasil = predict_stock(model_b, df_m_full['Close'].values, lookback=12)
            st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Bulanan Lengkap (OHLCV)"):
            st.dataframe(df_m_full.sort_index(ascending=False), use_container_width=True)
