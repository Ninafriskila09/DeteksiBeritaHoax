import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix

# Memuat model dan vectorizer yang sudah disimpan
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')  # Pastikan Anda sudah menyimpan model yang dilatih

# Memuat data tambahan jika diperlukan
dataset = pd.read_excel('dataset_clean.xlsx')

def load_data():
    return dataset

def preprocess_data(data):
    X_raw = data["clean_text"]
    y_raw = data["Label"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_TFIDF = vectorizer.fit_transform(X_raw)

    return X_TFIDF, y_raw, vectorizer

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 style='text-align: center;'>Sistem Deteksi Berita Hoax Naive Bayes</h2>",
                unsafe_allow_html=True)

    menu = st.sidebar.radio("Pilih Menu", ["Deteksi Berita", "Evaluasi Model", "Visualisasi Word Cloud"])

    data = load_data()
    X_features, y_labels, vectorizer = preprocess_data(data)

    if menu == "Deteksi Berita":
        st.markdown("**Masukkan Kalimat untuk Prediksi**")
        input_text = st.text_area("", height=150)

        detect_button = st.button("Deteksi")

        if detect_button and input_text:
            # Transformasi teks dengan vectorizer yang sama dengan yang digunakan untuk melatih model
            input_text_tfidf = vectorizer.transform([input_text])
            input_text_dense = csr_matrix.toarray(input_text_tfidf)

            # Prediksi menggunakan model
            prediction_proba = model.predict_proba(input_text_dense)
            proba_fakta = prediction_proba[0][1]  # Probabilitas untuk kelas 'Fakta'
            proba_hoax = prediction_proba[0][0]   # Probabilitas untuk kelas 'Hoax'

            # Menampilkan hasil
            sentiment = "Fakta" if proba_fakta > proba_hoax else "Hoax"
            color = "green" if sentiment == "Fakta" else "red"

            st.markdown(f"""
    <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
        <strong>{sentiment}</strong><br>
        Probabilitas Fakta: {proba_fakta * 100:.2f}%<br>
        Probabilitas Hoax: {proba_hoax * 100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    elif menu == "Evaluasi Model":
        # Memisahkan data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        # Evaluasi model
        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_metrics(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        display_wordclouds(data)

if __name__ == '__main__':
    main()
