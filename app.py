import streamlit as st
import pandas as pd
import plotly.express as px
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Sentimen PSU", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI yang bersih dan profesional
st.markdown("""
    <style>
    .main { background-color: #F4F6F9; }
    h1, h2, h3 { color: #0F172A; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 16px;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Utama Dashboard
st.title("🤖 Opini Publik terhadap Pemilihan Suara Ulang di Media Sosial X")
st.markdown("---")

# --- 2. FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_psu (4).csv', delimiter=',', on_bad_lines='skip', lineterminator='\n', quoting=csv.QUOTE_NONE)
        df.columns = df.columns.str.replace('"', '').str.replace(';', '').str.strip()
        
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        if 'full_text' in df.columns:
            df['word_count'] = df['full_text'].astype(str).apply(lambda x: len(x.split()))
            
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# Pengaturan Urutan Kategori Sesuai Gambar Anda
kategori_lengkap = ['Negatif', 'Netral', 'Positif']
warna_custom = {'Positif': '#10B981', 'Negatif': '#EF4444', 'Netral': '#94A3B8'}

# --- 3. SIDEBAR INFORMASI TEKNIS ---
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWebVIzlwfEBB4xt6wSIRFFGMFRPhsemOcxN40d3f1Lw&s=10", width=120)
    st.markdown("### Profil Sistem")
    st.info("""
    **Metodologi Ekstraksi & Model:**
    - **Sumber Data:** X (Twitter)
    - **Algoritma Utama:** CNN (Convolutional Neural Networks)
    - **Pelabelan:** Kombinasi Leksikon & Validasi Manual
    """)
    st.markdown("---")
    st.caption("Dashboard v2.6")

# --- 4. LOGIKA HALAMAN DENGAN TABS ---
if not df.empty and 'sentiment_name' in df.columns:
    sentimen_counts = df['sentiment_name'].value_counts().reindex(kategori_lengkap, fill_value=0)
    
    # Membuat 4 Tabs Utama secara berurutan dan sejajar
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan Eksekutif", 
        "☁️ Analisis Teks (WordCloud)", 
        "🎯 Evaluasi Model (CNN)", 
        "🔎 Filter & Ekspor Data"
    ])
    
    # ==========================================
    # TAB 1: RINGKASAN EKSEKUTIF (KPI & GRAFIK)
    # ==========================================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Unik", f"{len(df):,}", "Setelah Preprocessing")
        c2.metric("Negatif", f"{sentimen_counts['Negatif']:,}", "Opini Menolak", delta_color="inverse")
        c3.metric("Netral", f"{sentimen_counts['Netral']:,}", "Opini Objektif", delta_color="off")
        c4.metric("Positif", f"{sentimen_counts['Positif']:,}", "Opini Mendukung")
        
        st.write("<br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### Komposisi Sentimen Keseluruhan")
            fig_pie = px.pie(names=sentimen_counts.index, values=sentimen_counts.values, hole=0.5,
                             color=sentimen_counts.index, color_discrete_map=warna_custom)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            if 'created_at' in df.columns and not df['created_at'].isnull().all():
                st.markdown("#### Tren Sentimen Harian")
                df['Tanggal'] = df['created_at'].dt.date
                tren_df = df.groupby(['Tanggal', 'sentiment_name']).size().reset_index(name='Jumlah')
                fig_line = px.line(tren_df, x='Tanggal', y='Jumlah', color='sentiment_name', color_discrete_map=warna_custom, markers=True)
                fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.markdown("#### Distribusi Sentimen (Bar)")
                df_bar = sentimen_counts.reset_index().rename(columns={'index':'Sentimen', 'sentiment_name':'Jumlah'})
                fig_bar = px.bar(df_bar, x='Sentimen', y='Jumlah', color='Sentimen', color_discrete_map=warna_custom, text_auto=True)
                fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # TAB 2: ANALISIS TEKS LENGKAP (NLP)
    # ==========================================
    with tab2:
        st.markdown("#### Ekstraksi Topik Kata (Word Cloud)")
        pilihan_wc = st.selectbox("Pilih Sentimen untuk dianalisis dominasi katanya:", ["Positif", "Negatif", "Netral"])
        
        teks_gabungan = " ".join(df[df['sentiment_name'] == pilihan_wc]['full_text'].astype(str).tolist())
        
        if teks_gabungan.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(teks_gabungan)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Tidak ada kata yang cukup untuk membentuk Word Cloud pada sentimen ini.")

        st.markdown("---")
        st.markdown("#### Distribusi Panjang Teks (Word Count)")
        if 'word_count' in df.columns:
            fig_hist = px.histogram(df, x="word_count", color="sentiment_name", nbins=50, 
                                    color_discrete_map=warna_custom, barmode="overlay")
            fig_hist.update_layout(xaxis_title="Jumlah Kata dalam Satu Tweet", yaxis_title="Frekuensi")
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==========================================
    # TAB 3: EVALUASI MODEL (CONFUSION MATRIX)
    # ==========================================
    with tab3:
        st.header("Metrik Evaluasi dan Validasi Model CNN")
        st.markdown("Halaman ini menyajikan performa model CNN berdasarkan hasil pengujian sistem.")
        
        st.info("### Akurasi Keseluruhan Sistem: **92.20%**")
        st.write("<br>", unsafe_allow_html=True)
        
        col_eval1, col_eval2 = st.columns([4, 5])
        
        with col_eval1:
            st.markdown("#### Confusion Matrix")
            matrix_data = [
                [2, 5, 0],    
                [1, 155, 0],  
                [0, 10, 32]   
            ]
            
            fig_cm = px.imshow(
                matrix_data,
                x=kategori_lengkap,
                y=kategori_lengkap,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Kelas Prediksi", y="Kelas Aktual", color="Jumlah Data")
            )
            fig_cm.update_layout(
                xaxis=dict(title='Kelas Prediksi'),
                yaxis=dict(title='Kelas Aktual'),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_eval2:
            st.markdown("#### Classification Report")
            report_data = {
                'Kelas Sentimen': ['Negatif', 'Netral', 'Positif', 'accuracy', 'macro avg', 'weighted avg'],
                'Precision': [0.67, 0.91, 1.00, '', 0.86, 0.92],
                'Recall': [0.29, 0.99, 0.76, '', 0.68, 0.92],
                'F1-Score': [0.40, 0.95, 0.86, 0.92, 0.74, 0.91],
                'Support': [7, 156, 42, 205, 205, 205]
            }
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Analisis Singkat Performa:**
            - Model bekerja sangat baik pada kelas **Netral** dengan F1-Score mencapai **0.95**.
            - Untuk kelas **Positif**, tingkat kepastian (*Precision*) mencapai **1.00** (Sempurna), artinya tidak ada opini negatif/netral yang salah dikenali sebagai positif.
            - Total data uji (*testing support*) yang digunakan untuk validasi ini berjumlah **205 tweet**.
            """)

    # ==========================================
    # TAB 4: FILTER & EKSPOR DATA
    # ==========================================
    with tab4:
        st.markdown("#### Database Tweet Interaktif")
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filter_s = st.selectbox("Filter Sentimen:", ["Semua", "Positif", "Negatif", "Netral"])
        with col_f2:
            search_kw = st.text_input("Pencarian Teks (Kata Kunci):")

        df_tampil = df.copy()
        if filter_s != "Semua":
            df_tampil = df_tampil[df_tampil['sentiment_name'] == filter_s]
        if search_kw:
            df_tampil = df_tampil[df_tampil['full_text'].astype(str).str.contains(search_kw, case=False, na=False)]

        st.success(f"Menampilkan {len(df_tampil):,} baris data.")
        
        kolom_tampil = [k for k in ['created_at', 'username', 'full_text', 'sentiment_name', 'word_count'] if k in df.columns]
        st.dataframe(df_tampil[kolom_tampil], use_container_width=True, height=350)

        csv_data = df_tampil.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data yang Difilter (CSV)", data=csv_data, file_name='dataset_filtered.csv', mime='text/csv')

elif df.empty:
    st.error("⚠️ Dataset tidak ditemukan. Pastikan file 'data_psu (4).csv' berada di direktori yang sama.")
else:
    st.error("⚠️ Kolom 'sentiment_name' tidak ditemukan. Pastikan proses pelabelan data sudah tersimpan di file CSV.")