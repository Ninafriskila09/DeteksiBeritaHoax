import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from scipy.sparse import csr_matrix

# Memuat model dan vectorizer yang sudah disimpan
vectorizer = joblib.load('vectorizer.pkl')

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

def train_model(X_train, y_train):
    NB = GaussianNB()
    X_train_dense = csr_matrix.toarray(X_train)
    NB.fit(X_train_dense, y_train)
    return NB

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Untuk analisis berita baru
def classify_news(news):
    news_transformed = vectorizer.transform([news])
    prediction = model.predict(news_transformed)
    return prediction

# Contoh berita untuk analisis
news_article = "Scientists find new evidence about climate change"
prediction = classify_news(news_article)
print(f"Prediction for news: {'Hoax' if prediction[0] == 1 else 'Fact'}")

# Menghitung persentase kata
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

# Analisis berita
hoax_percentage, fact_percentage = calculate_word_percentage(news_article, model, vectorizer)
print(f"Hoax words percentage: {hoax_percentage:.2f}%")
print(f"Fact words percentage: {fact_percentage:.2f}%")

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
    # Mengubah background menjadi putih dengan CSS
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

    # Sidebar menu
    menu = st.sidebar.radio("Pilih Menu", ["Deteksi Berita", "Evaluasi Model", "Visualisasi Word Cloud"])

    # Load data dan preprocess
    data = load_data()
    X_features, y_labels, vectorizer = preprocess_data(data)

    if menu == "Deteksi Berita":
        st.markdown("**Masukkan Judul Prediksi**")
        input_text = st.text_area("", height=150)

        detect_button = st.button("Deteksi")

        if detect_button and input_text:
            # Memisahkan data untuk pelatihan dan pengujian
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)

            # Transformasi teks dengan vectorizer yang digunakan untuk melatih model
            input_text_tfidf = vectorizer.transform([input_text])
            input_text_dense = csr_matrix.toarray(input_text_tfidf)

            # Prediksi menggunakan model yang telah dimuat
            prediction = model.predict(input_text_dense)
            sentiment = "Fakta" if prediction[0] == 0 else "Hoax"

            # Menampilkan hasil
            color = "green" if sentiment == "Fakta" else "red"
            st.markdown(f"""
    <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
        <strong>{sentiment}</strong>
    </div>
    """, unsafe_allow_html=True)
            
    elif menu == "Evaluasi Model":
        # Memisahkan data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        # Evaluasi model
        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_evaluation(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        # Tampilkan Word Cloud di bawah hasil
        display_wordclouds(data)

if __name__ == '__main__':
    main()
