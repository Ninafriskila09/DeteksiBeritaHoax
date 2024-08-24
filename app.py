import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    model = MultinomialNB()
    X_train_dense = csr_matrix.toarray(X_train)
    model.fit(X_train_dense, y_train)
    return model

def display_metrics(y_test, y_pred):
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.text(classification_report(y_test, y_pred))

    st.subheader('Confusion Matrix')
    confm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(confm, index=['0', '1'], columns=['0', '1'])

    fig, ax = plt.subplots()
    sns.heatmap(df_cm, cmap='Greens', annot=True, fmt=".0f", ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('True Sentiment')

    st.pyplot(fig)

def generate_sample_data():
    np.random.seed(0)
    y_test = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    return y_test, y_pred

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
    X_features, y_labels, vectorizer = preprocess_data(data)

    if menu == "Deteksi Berita":
        st.markdown("**Masukkan Judul Prediksi**")
        input_text = st.text_area("", height=150)

        detect_button = st.button("Deteksi")

        if detect_button and input_text:
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)

            input_text_tfidf = vectorizer.transform([input_text])
            input_text_dense = csr_matrix.toarray(input_text_tfidf)

            # Get prediction and probability
            prediction = model.predict(input_text_dense)
            prediction_proba = model.predict_proba(input_text_dense)
            sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
            sentiment_proba = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

            color = "green" if sentiment == "Fakta" else "red"
            st.markdown(f"""
    <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
        <strong>{sentiment}</strong><br>
        Probabilitas: {sentiment_proba * 100:.2f}%
    </div>
    """, unsafe_allow_html=True)
            
    elif menu == "Evaluasi Model":
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_metrics(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        display_wordclouds(data)

if __name__ == '__main__':
    main()
