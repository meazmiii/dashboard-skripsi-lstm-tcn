import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Judul & Deskripsi
st.set_page_config(page_title="Dashboard Skripsi - LSTM vs TCN", layout="wide")
st.title("📊 Analisis Komparasi LSTM vs TCN")
st.markdown("Studi kasus: Prediksi Harga Saham **BBCA.JK**")

# 2. Load Data (Otomatis membaca file CSV)
try:
    df = pd.read_csv('Hasil_Eksperimen_Tuning.csv')
    
    # Pastikan urutan Timeframe benar
    urutan_custom = ['Harian', 'Mingguan', 'Bulanan']
    df['Time Frame'] = pd.Categorical(df['Time Frame'], categories=urutan_custom, ordered=True)
    df = df.sort_values(by=['Time Frame', 'RMSE'])
    
    # Sidebar (Filter Pilihan)
    st.sidebar.header("Filter Data")
    timeframe_pilihan = st.sidebar.multiselect(
        "Pilih Timeframe:",
        options=df['Time Frame'].unique(),
        default=df['Time Frame'].unique()
    )
    
    # Filter DataFrame berdasarkan pilihan
    df_filtered = df[df['Time Frame'].isin(timeframe_pilihan)]

    # 3. Tampilkan Grafik Utama (Best Model)
    st.subheader("🏆 Perbandingan Performa Model Terbaik")
    
    # Ambil juara 1 per model per timeframe
    best_per_model = df_filtered.groupby(['Time Frame', 'Model'], observed=True).apply(lambda x: x.nsmallest(1, 'RMSE')).reset_index(drop=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=best_per_model, x='Time Frame', y='RMSE', hue='Model', ax=ax, palette='muted')
        ax.set_title('RMSE Terkecil (Lebih Rendah Lebih Baik)')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
    with col2:
        st.write("### Data Detail (Top 5)")
        st.dataframe(df_filtered.sort_values('RMSE').head(5)[['Time Frame', 'Model', 'RMSE', 'Units', 'LR']])

    # 4. Tampilkan Tabel Lengkap
    st.markdown("---")
    st.subheader("📋 Data Lengkap Hasil Eksperimen")
    st.dataframe(df_filtered)

except FileNotFoundError:
    st.error("File 'Hasil_Eksperimen_Tuning.csv' tidak ditemukan. Pastikan file sudah diupload ke GitHub.")