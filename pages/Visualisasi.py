import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Shovia - Visualisasi", page_icon="🎥", layout="wide")
st.title("📈 Visualisasi & Analisis Sentimen")

# ==========================================
# 1. VERIFIKASI DATA DARI SESSION STATE
# ==========================================
# Mengecek apakah user sudah melakukan proses klasifikasi di halaman sebelumnya
if 'df_final_classified' not in st.session_state or st.session_state['df_final_classified'] is None:
    st.warning("⚠️ Belum ada data yang diklasifikasi.")
    st.info("Silakan buka menu **'Konfigurasi'** terlebih dahulu untuk melakukan scraping dan klasifikasi sentimen.")
    st.stop() # Menghentikan eksekusi kode di bawah jika data belum ada

# Mengambil data dari memori (session state)
df = st.session_state['df_final_classified'].copy()

# ==========================================
# 2. RINGKASAN METRIK (Quick Stats)
# ==========================================
total_data = len(df)
positif_count = len(df[df['sentimen'] == 'Positif'])
negatif_count = len(df[df['sentimen'] == 'Negatif'])

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Video Dianalisis", total_data)
col_m2.metric("Sentimen Positif", positif_count, delta=f"{(positif_count/total_data)*100:.1f}%", delta_color="normal")
col_m3.metric("Sentimen Negatif", negatif_count, delta=f"{(negatif_count/total_data)*100:.1f}%", delta_color="inverse")
st.divider()

# ==========================================
# 2.5 STATISTIK ENGAGEMENT (INTERAKSI)
# ==========================================

# Memastikan kolom engagement bertipe numerik dan mengisi nilai kosong (NaN) dengan 0
df['videoViewCount'] = pd.to_numeric(df['videoViewCount'], errors='coerce').fillna(0)
df['likesCount'] = pd.to_numeric(df['likesCount'], errors='coerce').fillna(0)
df['commentsCount'] = pd.to_numeric(df['commentsCount'], errors='coerce').fillna(0)

# Menghitung total masing-masing interaksi
total_views = int(df['videoViewCount'].sum())
total_likes = int(df['likesCount'].sum())
total_comments = int(df['commentsCount'].sum())

# Fungsi untuk memformat angka agar lebih rapi (misal: 1500000 -> 1.500.000)
def format_angka(angka):
    return f"{angka:,}".replace(",", ".")

# Menampilkan metrik engagement dalam 3 kolom
col_e1, col_e2, col_e3 = st.columns(3)
col_e1.metric("👀 Total Views", format_angka(total_views))
col_e2.metric("❤️ Total Likes", format_angka(total_likes))
col_e3.metric("💬 Total Comments", format_angka(total_comments))

st.divider()

# ==========================================
# 3. VISUALISASI PIE CHART
# ==========================================
st.subheader("Proporsi Kelas Sentimen")

# Menyiapkan data untuk Pie Chart
df_pie = df['sentimen'].value_counts().reset_index()
df_pie.columns = ['Sentimen', 'Jumlah']

# Membuat Pie Chart menggunakan Plotly
fig = px.pie(
    df_pie, 
    values='Jumlah', 
    names='Sentimen',
    hole=0.4, # Membuat Donut Chart agar lebih modern
    color='Sentimen',
    color_discrete_map={'Positif': '#4CAF50', 'Negatif': '#F44336', 'Netral': '#9E9E9E'},
    category_orders={"Sentimen": ["Positif", "Negatif", "Netral"]}
)

# Mengatur tampilan layout grafik
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(showlegend=True)

# Menampilkan grafik di Streamlit
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. VISUALISASI N-GRAMS DENGAN CUSTOM STOPWORDS
# ==========================================
st.divider()
st.subheader("Analisis N-Gram (Top Kosakata)")
st.write("Visualisasi kata dan frasa yang paling sering muncul berdasarkan kelas sentimen.")

# Menyiapkan list default stopwords dari kode Anda
default_stopwords = [
    'yg', 'ya', 'kan', 'nih', 'tuh', 'sih', 'aja', 'dong', 'gue', 'gua', 'lu', 'lo',
    'kalo', 'kalau', 'dan', 'di', 'ini', 'itu', 'untuk', 'dari', 'yang', 'ada',
    'kita', 'kami', 'saya', 'dia', 'mereka', 'buat', 'terus', 'jadi', 'sudah', 'udah',
    'belum', 'bisa', 'sama', 'juga', 'lebih', 'sangat', 'dengan', 'karena', 'kok',
    'kayak', 'gitu', 'gini', 'pun', 'pas', 'deh', 'nah', 'banget', 'cuma',
    'dalam', 'pada', 'hal', 'apa', 'gimana', 'bagaimana', 'kenapa', 'atau', 'maka',
    'gak', 'nggak', 'bukan', 'memang', 'baru', 'sekarang', 'sampai', 'harus', 'semua',
    'orang', 'banyak', 'hari', 'masuk', 'bikin', 'kasih', 'dulu', 'lagi', 'lah', 'pak',
    'mau', 'jangan', 'misal', 'bilang', 'akhir', 'tahu', 'tau', 'mungkin', 
    'kalian', 'satu', 'nama', 'nya', 'biar', 'soal', 'paling', 
    'benar', 'segala', 'macem', 'kaya', 'oke', 'tadi', 'sini', 'kata', 
    'arti', 'lihat', 'bahkan', 'nyata', 'coba', 'beri', 'dapat', 'iya', 'gin',
    'paking', 'pagi', 'hai', 'yakan'
]

# Text area untuk input custom stopwords oleh user (dipisah dengan koma)
st.write("**⚙️ Pengaturan Stopwords**")
user_stopwords_input = st.text_area(
    "Tambahkan atau kurangi kata yang tidak ingin ditampilkan (pisahkan dengan koma):", 
    value=", ".join(default_stopwords),
    height=150
)

# Tombol untuk mengeksekusi pembuatan N-Gram (agar tidak me-render ulang setiap kali user mengetik)
if st.button("📊 Tampilkan Grafik N-Gram", type="primary"):
    with st.spinner("Mengekstrak dan memetakan N-Grams..."):
        # Memproses input stopwords dari user menjadi list yang bersih
        final_stopwords = [word.strip().lower() for word in user_stopwords_input.split(',')]
        
        # Pastikan kolom clean_transcript ada (dari hasil NLP halaman sebelumnya)
        teks_kolom = 'clean_transcript' if 'clean_transcript' in df.columns else 'transcript'
        
        # Mendapatkan sentimen unik yang ada di data (misal: Positif dan Negatif)
        kelas_sentimen = df['sentimen'].unique()
        
        # Pengaturan Tema Seaborn
        sns.set_theme(style="whitegrid")
        
        def get_top_ngrams(corpus, n_top=10, ngram_range=(1,1)):
            try:
                vec = CountVectorizer(stop_words=final_stopwords, ngram_range=ngram_range).fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                return words_freq[:n_top]
            except ValueError:
                # Terjadi jika corpus kosong atau semua kata terbuang oleh stopwords
                return []

        ngram_types = {'Unigram (1 Kata)': (1,1), 'Bigram (2 Kata)': (2,2), 'Trigram (3 Kata)': (3,3)}
        
        # Membuat Tabs untuk memisahkan Unigram, Bigram, Trigram agar UI rapi
        tabs = st.tabs(list(ngram_types.keys()))
        
        for tab, (ngram_name, ngram_range) in zip(tabs, ngram_types.items()):
            with tab:
                # Menyiapkan kanvas matplotlib secara dinamis berdasarkan jumlah kelas sentimen
                fig, axes = plt.subplots(1, len(kelas_sentimen), figsize=(6 * len(kelas_sentimen), 5))
                
                # Jika hanya ada 1 kelas sentimen, axes tidak berupa array. Kita jadikan list agar tidak error.
                if len(kelas_sentimen) == 1: axes = [axes]
                
                for i, sentiment in enumerate(kelas_sentimen):
                    # Mengambil teks khusus untuk sentimen tersebut
                    corpus = df[df['sentimen'] == sentiment][teks_kolom].fillna('')
                    
                    top_ngrams = get_top_ngrams(corpus, n_top=10, ngram_range=ngram_range)
                    
                    if top_ngrams:
                        df_plot = pd.DataFrame(top_ngrams, columns=['N-gram', 'Frekuensi'])
                        
                        # Plot menggunakan seaborn (orient='h' agar horizontal)
                        sns.barplot(data=df_plot, x='Frekuensi', y='N-gram', ax=axes[i], 
                                    color='#1f77b4', legend=False)
                        
                        axes[i].set_title(f'Sentimen: {sentiment}', fontsize=12, fontweight='bold')
                        axes[i].set_xlabel('Frekuensi')
                        axes[i].set_ylabel('')
                    else:
                        axes[i].set_title(f'Sentimen: {sentiment}\n(Tidak ada kata yang valid)', fontsize=12)
                        axes[i].axis('off')
                        
                plt.tight_layout()
                # Menampilkan plot di Streamlit
                st.pyplot(fig)

# ==========================================
# 5. TABEL HASIL AKHIR
# ==========================================
st.divider()
st.subheader("📋 Tabel Hasil Klasifikasi")

# Menambahkan kolom nomor urut (start from 1)
df.insert(0, 'No.', range(1, 1 + len(df)))

# Menata ulang urutan kolom agar kolom Sentimen mudah terlihat (paling depan setelah No.)
cols = list(df.columns)
# Memindahkan kolom 'sentimen' ke urutan kedua setelah 'No.'
if 'sentimen' in cols:
    cols.insert(1, cols.pop(cols.index('sentimen')))
df_display = df[cols]

# Menampilkan Tabel Final
st.dataframe(
    df_display, 
    hide_index=True, 
    use_container_width=True
)

# ==========================================
# 6. FITUR DOWNLOAD (Opsional & Berguna)
# ==========================================
st.write("")
csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Hasil Analisis (.csv)",
    data=csv,
    file_name=f"hasil_analisis_shovia_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)
