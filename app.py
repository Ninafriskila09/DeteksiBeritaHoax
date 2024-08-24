import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Memuat model dan vectorizer yang sudah disimpan
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('multinomial_nb_model.pkl')

# Memuat data tambahan jika diperlukan
dataset = pd.read_excel('dataset_clean.xlsx')

def load_data():
    return dataset

def preprocess_data(data, vectorizer):
    X_raw = data["clean_text"]
    y_raw = data["Label"]
    X_TFIDF = vectorizer.transform(X_raw)
    return X_TFIDF, y_raw

def display_evaluation(y_test, y_pred):
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))
    columns = sorted(y_test.unique())
    confm = confusion_matrix(y_test, y_pred, labels=columns)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)
    st.write("**Confusion Matrix:**")
    st.write(df_cm)

def display_wordclouds(data):
    st.write("**Word Cloud untuk Semua Data:**")
    all_text = ' '.join(data['clean_text'])
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), use_column_width=True)

    st.write("**Word Cloud untuk Fakta:**")
    fakta = data[data['Label'] == 1]
    all_text_fakta = ' '.join(fakta['clean_text'])
    wordcloud_fakta = WordCloud(width=800, height=400, background_color='white').generate(all_text_fakta)
    st.image(wordcloud_fakta.to_array(), use_column_width=True)

    st.write("**Word Cloud untuk Hoax:**")
    hoax = data[data['Label'] == 0]
    all_text_hoax = ' '.join(hoax['clean_text'])
    wordcloud_hoax = WordCloud(width=800, height=400, background_color='white').generate(all_text_hoax)
    st.image(wordcloud_hoax.to_array(), use_column_width=True)

def plot_probabilities(probabilities):
    plt.bar(['Hoax', 'Fakta'], probabilities)
    plt.xlabel('Kelas')
    plt.ylabel('Probabilitas')
    plt.title('Probabilitas Prediksi')
    st.pyplot(plt)

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

    st.markdown("<h2 style='text-align: center;'>Sistem Deteksi Berita Hoax Naive Bayes
