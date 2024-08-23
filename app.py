import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
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

def display_evaluation(y_test, y_pred):
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    columns = sorted(y_test.unique())
    confm = confusion_matrix(y_test, y_pred, labels=columns)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)

    st.write("Confusion Matrix:")
    st.write(df_cm)

def display_wordclouds(data):
    st.write("Word Cloud untuk Semua Data:")
    all_text = ' '.join(data['clean_text'])
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), use_column_width=True)

    st.write("Word Cloud untuk Fakta:")
    fakta = data[data['Label'] == 1]
    all_text_fakta = ' '.join(fakta['clean_text'])
    wordcloud_fakta = WordCloud(width=800, height=400, background_color='white').generate(all_text_fakta)
    st.image(wordcloud_fakta.to_array(), use_column_width=True)

    st.write("Word Cloud untuk Hoax:")
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
        .fakta-text {
            color: green;
        }
        .hoax-text {
            color: red;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
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
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 24px; font-weight: bold;'>Masukkan Judul Prediksi</h3>",
                    unsafe_allow_html=True)

        st.markdown("<div class='centered'>", unsafe_allow_html=True)
        input_text = st.text_area("", height=120)
        detect_button = st.button("Deteksi")
        st.markdown("</div>", unsafe_allow_html=True)

        if detect_button and input_text:
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)

            input_text_tfidf = vectorizer.transform([input_text])
            input_text_dense = csr_matrix.toarray(input_text_tfidf)

            prediction = model.predict(input_text_dense)
            sentiment = "Fakta" if prediction[0] == 1 else "Hoax"

            st.markdown("<div class='centered'>", unsafe_allow_html=True)
            if sentiment == "Fakta":
                st.markdown(f"<h3 class='fakta-text'>{sentiment}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 class='hoax-text'>{sentiment}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            total_count = len(data)
            fact_count = data[data['Label'] == 1].shape[0]
            hoax_count = data[data['Label'] == 0].shape[0]

            fact_percentage = (fact_count / total_count) * 100
            hoax_percentage = (hoax_count / total_count) * 100

            st.markdown("<div class='centered'>", unsafe_allow_html=True)
            st.markdown(f"<p class='fakta-text'>Persentase Fakta: {fact_percentage:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='hoax-text'>Persentase Hoax: {hoax_percentage:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif menu == "Evaluasi Model":
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_evaluation(y_test, y_pred)

    elif menu == "Visualisasi Word Cloud":
        display_wordclouds(data)

if __name__ == '__main__':
    main()
