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


def main():
    st.title("Aplikasi Klasifikasi Sentimen")
     # Input teks untuk diprediksi
        st.write("Masukkan teks untuk diprediksi:")
        input_text = st.text_input("Teks", "")
        if input_text:
            input_




if __name__ == '__main__':
    main()
