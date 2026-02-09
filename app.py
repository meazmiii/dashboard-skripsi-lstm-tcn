import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from streamlit_autorefresh import st_autorefresh

# 1. Konfigurasi Halaman & Auto-refresh (1 detik)
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

# 3. Fungsi Prediksi Universal
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

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian (LSTM)")
        last_p = float(close_series.iloc[-1])
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, lookback=60)

        c1, c2 = st.columns(2)
        with c1: 
            st.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}")
            st.caption(f"Status: Real-time WIB")
        with c2: 
            st.metric("Prediksi LSTM", f"Rp {pred_today:,.2f}")
        
        st.markdown("---")
        # FITUR KEMBALI: Historis Akurasi 5 Hari
        st.write("### 🕒 Historis Akurasi (5 Hari Bursa Terakhir)")
        history_h = []
        for i in range(1, 6):
            t_idx = -i
            act_val = close_series.iloc[t_idx]
            act_date = close_series.index[t_idx].date()
            p_val = predict_stock(model_h, close_series.iloc[:t_idx].values, lookback=60)
            history_h.append({
                "Tanggal": act_date,
                "Harga Aktual": f"Rp {act_val:,.2f}",
                "Prediksi LSTM": f"Rp {p_val:,.2f}",
                "Selisih (Rp)": f"{abs(act_val - p_val):,.2f}"
            })
        st.table(pd.DataFrame(history_h))

        # FITUR KEMBALI: Tombol Prediksi Besok
        if st.button('Jalankan Prediksi LSTM (Besok)'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_h, close_series.values, lookback=60)
                st.success(f"### Estimasi Harga LSTM Besok: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Harian Lengkap (OHLCV)"):
            st.dataframe(df_all.sort_index(ascending=False), use_container_width=True)

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan (TCN)")
        df_w_full = df_all.resample('W-MON').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_w = float(df_w_full['Close'].iloc[-1])
        pred_w = predict_stock(model_m, df_w_full['Close'].values[:-1], lookback=24)

        col1, col2 = st.columns(2)
        with col1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
        with col2: st.metric("Prediksi TCN (Minggu Ini)", f"Rp {pred_w:,.2f}")

        st.markdown("---")
        # Historis Akurasi 5 Minggu Terakhir
        st.write("### 🕒 Historis Akurasi (5 Minggu Terakhir)")
        history_w = []
        for i in range(1, 6):
            t_idx = -i
            act_val = df_w_full['Close'].iloc[t_idx]
            act_date = df_w_full.index[t_idx].date()
            p_val = predict_stock(model_m, df_w_full['Close'].values[:t_idx], lookback=24)
            history_w.append({"Tanggal": act_date, "Harga Aktual": f"Rp {act_val:,.2f}", "Prediksi TCN": f"Rp {p_val:,.2f}", "Selisih (Rp)": f"{abs(act_val - p_val):,.2f}"})
        st.table(pd.DataFrame(history_w))

        if st.button('Jalankan Prediksi TCN (Minggu Depan)'):
            hasil = predict_stock(model_m, df_w_full['Close'].values, lookback=24)
            st.success(f"### Estimasi Harga TCN Minggu Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Mingguan Lengkap (OHLCV)"):
            st.dataframe(df_w_full.sort_index(ascending=False), use_container_width=True)

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan (TCN)")
        df_m_full = df_all.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_m = float(df_m_full['Close'].iloc[-1])
        pred_m = predict_stock(model_b, df_m_full['Close'].values[:-1], lookback=12)

        k1, k2 = st.columns(2)
        with k1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
        with k2: st.metric("Prediksi TCN (Bulan Ini)", f"Rp {pred_m:,.2f}")

        st.markdown("---")
        # Historis Akurasi 5 Bulan Terakhir
        st.write("### 🕒 Historis Akurasi (5 Bulan Terakhir)")
        history_m = []
        for i in range(1, 6):
            t_idx = -i
            act_val = df_m_full['Close'].iloc[t_idx]
            act_date = df_m_full.index[t_idx].date()
            p_val = predict_stock(model_b, df_m_full['Close'].values[:t_idx], lookback=12)
            history_m.append({"Tanggal": act_date, "Harga Aktual": f"Rp {act_val:,.2f}", "Prediksi TCN": f"Rp {p_val:,.2f}", "Selisih (Rp)": f"{abs(act_val - p_val):,.2f}"})
        st.table(pd.DataFrame(history_m))

        if st.button('Jalankan Prediksi TCN (Bulan Depan)'):
            hasil = predict_stock(model_b, df_m_full['Close'].values, lookback=12)
            st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Bulanan Lengkap (OHLCV)"):
            st.dataframe(df_m_full.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
