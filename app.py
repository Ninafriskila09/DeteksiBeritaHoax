import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
    model = MultinomialNB()  # Mengganti GaussianNB dengan MultinomialNB
    model.fit(X_train, y_train)
    return model

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
    st.markdown("**Masukkan Judul untuk Prediksi**")
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

        # Menghitung probabilitas untuk setiap kelas
        probabilities = model.predict_proba(input_text_dense)
        prob_fakta = probabilities[0][1] * 100
        prob_hoax = probabilities[0][0] * 100

        # Menampilkan hasil
        sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
        color = "green" if sentiment == "Fakta" else "red"

        st.markdown(f"""
<div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
    <strong>{sentiment}</strong>
</div>
""", unsafe_allow_html=True)

        st.write(f"**Probabilitas Fakta:** {prob_fakta:.2f}%")
        st.write(f"**Probabilitas Hoax:** {prob_hoax:.2f}%")


    elif menu == "Evaluasi Model":
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        display_evaluation(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        display_wordclouds(data)

if __name__ == '__main__':
    main()
