import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

# Memuat model dan vectorizer yang sudah disimpan
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')  # Pastikan model juga dimuat jika sudah disimpan

# Memuat data tambahan jika diperlukan
dataset = pd.read_excel('dataset_clean.xlsx')

def load_data():
    return dataset

def preprocess_data(data):
    X_raw = data["clean_text"]
    y_raw = data["Label"]
    X_TFIDF = vectorizer.transform(X_raw)  # Gunakan vectorizer yang sudah dimuat
    return X_TFIDF, y_raw

def display_evaluation(y_test, y_pred):
    st.write("**Evaluation Report:**")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix:**")
    st.write(cm)

def calculate_word_percentage(news, model, vectorizer):
    words = news.split()
    word_vectors = vectorizer.transform(words)
    predictions = model.predict(word_vectors)
    num_words = len(words)
    num_hoax_words = np.sum(predictions)
    num_fact_words = num_words - num_hoax_words
    hoax_percentage = (num_hoax_words / num_words) * 100
    fact_percentage = (num_fact_words / num_words) * 100
    return hoax_percentage, fact_percentage

def display_wordclouds(data):
    st.write("**Word Cloud untuk Semua Data:**")
    all_text = ' '.join(data['clean_text'])
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.write("**Word Cloud untuk Fakta:**")
    fakta = data[data['Label'] == 1]
    all_text_fakta = ' '.join(fakta['clean_text'])
    wordcloud_fakta = WordCloud(width=800, height=400, background_color='white').generate(all_text_fakta)
    st.image(wordcloud_fakta.to_array(), use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.write("**Word Cloud untuk Hoax:**")
    hoax = data[data['Label'] == 0]
    all_text_hoax = ' '.join(hoax['clean_text'])
    wordcloud_hoax = WordCloud(width=800, height=400, background_color='white').generate(all_text_hoax)
    st.image(wordcloud_hoax.to_array(), use_column_width=True)

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
    X_features, y_labels = preprocess_data(data)

    if menu == "Deteksi Berita":
        st.markdown("**Masukkan Judul Prediksi**")
        input_text = st.text_area("", height=150)

        detect_button = st.button("Deteksi")

        if detect_button and input_text:
            input_text_tfidf = vectorizer.transform([input_text])
            prediction = model.predict(input_text_tfidf)
            sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
            color = "green" if sentiment == "Fakta" else "red"
            st.markdown(f"""
    <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
        <strong>{sentiment}</strong>
    </div>
    """, unsafe_allow_html=True)
            
    elif menu == "Evaluasi Model":
        # Menggunakan model yang dimuat dari file
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        display_evaluation(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        display_wordclouds(data)

if __name__ == '__main__':
    main()
