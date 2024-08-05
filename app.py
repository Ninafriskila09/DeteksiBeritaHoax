import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import csr_matrix
from wordcloud import WordCloud

# Fungsi untuk membaca data dan melakukan preprocessing
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Fungsi untuk melakukan pemrosesan data
def preprocess_data(data):
    X_raw = data["clean_text"]
    y_raw = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(X_train)
    X_train_TFIDF = vectorizer.transform(X_train)
    X_test_TFIDF = vectorizer.transform(X_test)

    chi2_features = SelectKBest(chi2, k=500)
    X_kbest_features = chi2_features.fit_transform(X_train_TFIDF, y_train)

    return X_kbest_features, y_train, X_test_TFIDF, y_test, vectorizer, chi2_features

# Fungsi untuk melatih model
def train_model(X_train, y_train):
    NB = GaussianNB()
    # Mengonversi matriks sparse menjadi matriks padat
    X_train_dense = csr_matrix.toarray(X_train)
    NB.fit(X_train_dense, y_train)
    return NB

# Fungsi untuk menampilkan hasil evaluasi
def display_evaluation(y_test, y_pred):
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    columns = sorted(y_test.unique())
    confm = confusion_matrix(y_test, y_pred, labels=columns)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)

    st.write("Confusion Matrix:")
    st.write(df_cm)

# Fungsi untuk menampilkan wordcloud
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
    st.title("Aplikasi Klasifikasi Sentimen")

    # Upload file dataset
    st.write("Upload file dataset:")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        X_train, y_train, X_test, y_test, vectorizer, chi2_features = preprocess_data(data)
        model = train_model(X_train, y_train)

        # Input teks untuk diprediksi
        st.write("Masukkan teks untuk diprediksi:")
        input_text = st.text_input("Teks", "")
        if input_text:
            input_text_tfidf = vectorizer.transform([input_text])
            input_text_chi2 = chi2_features.transform(input_text_tfidf)
            input_text_chi2_dense = csr_matrix.toarray(input_text_chi2)
            prediction = model.predict(input_text_chi2_dense)
            sentiment = "Fakta" if prediction[0] == 1 else "Hoax"
            st.write("Hasil prediksi:", sentiment)

            # Evaluasi model
            X_test_chi2 = chi2_features.transform(X_test)
            X_test_chi2_dense = csr_matrix.toarray(X_test_chi2)
            y_pred = model.predict(X_test_chi2_dense)
            display_evaluation(y_test, y_pred)

            # Tampilkan Word Cloud
            display_wordclouds(data)

if __name__ == '__main__':
    main()
