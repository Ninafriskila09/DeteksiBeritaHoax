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

print(type(model))
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')


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

    st.markdown("<h2 style='text-align: center;'>Sistem Deteksi Berita Hoax Naive Bayes</h2>",
                unsafe_allow_html=True)

    # Sidebar menu
    menu = st.sidebar.radio("Pilih Menu", ["Deteksi Berita", "Evaluasi Model", "Visualisasi Word Cloud"])

    # Load data dan preprocess
    data = load_data()
    X_features, y_labels = preprocess_data(data, vectorizer)

    if menu == "Deteksi Berita":
        st.markdown("**Masukkan Judul Prediksi**")
        input_text = st.text_area("", height=150)
        detect_button = st.button("Deteksi")

        if detect_button and input_text:
            # Transformasi teks dengan vectorizer yang digunakan untuk melatih model
            input_text_tfidf = vectorizer.transform([input_text])
            # Debugging untuk memeriksa bentuk data yang diproses
            print(f"Input text TF-IDF shape: {input_text_tfidf.shape}")
            # Prediksi dan probabilitas menggunakan model yang telah dilatih
            if hasattr(model, 'predict_proba'):
                prediction_probabilities = model.predict_proba(input_text_tfidf)
                prediction = model.predict(input_text_tfidf)
                # Mendapatkan probabilitas untuk setiap kelas
                probability_fakta = prediction_probabilities[0][1]
                probability_hoax = prediction_probabilities[0][0]
                sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
                # Menampilkan hasil
                color = "green" if sentiment == "Fakta" else "red"
                st.markdown(f"""
                <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
                    <strong>{sentiment}</strong><br>
                    <span>Fakta: {probability_fakta * 100:.2f}%</span><br>
                    <span>Hoax: {probability_hoax * 100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                # Tampilkan grafik probabilitas
                plot_probabilities([probability_hoax, probability_fakta])
            else:
                # Jika model tidak memiliki predict_proba, gunakan predict saja
                prediction = model.predict(input_text_tfidf)
                sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
                color = "green" if sentiment == "Fakta" else "red"
                st.markdown(f"""
                <div style="text-align: center; background-color: {color}; color: white; padding: 10px;">
                    <strong>{sentiment}</strong><br>
                </div>
                """, unsafe_allow_html=True)

    elif menu == "Evaluasi Model":
        # Memisahkan data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        # Evaluasi model
        y_pred = model.predict(X_test)
        display_evaluation(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        # Tampilkan Word Cloud di bawah hasil
        display_wordclouds(data)

if __name__ == '__main__':
    main()
