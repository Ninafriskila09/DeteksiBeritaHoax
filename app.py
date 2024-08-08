import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, confusion_matrix

# Fungsi untuk memuat objek
def load_objects(vectorizer_filename, chi2_filename, model_filename):
    vectorizer = joblib.load(vectorizer_filename)
    chi2_features = joblib.load(chi2_filename)
    model = joblib.load(model_filename)
    return vectorizer, chi2_features, model

# Fungsi untuk memproses data baru
def preprocess_new_data(vectorizer, chi2_features, text_data):
    X_TFIDF = vectorizer.transform(text_data)
    X_kbest_features = chi2_features.transform(X_TFIDF)
    return X_kbest_features

# Fungsi untuk melatih model
def train_model(X_train, y_train):
    NB = GaussianNB()
    # Mengonversi matriks sparse menjadi matriks padat
    X_train_dense = csr_matrix.toarray(X_train)
    NB.fit(X_train_dense, y_train)
    return NB

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title('Text Classification App')

    # Muat objek yang telah disimpan
    vectorizer, chi2_features, model = load_objects('vectorizer.pkl', 'chi2_features.pkl', 'model.pkl')

    st.header('Enter Text for Classification')

    # Input teks dari pengguna
    user_input = st.text_area("Enter text here:", "")
    
    if st.button('Classify'):
        if user_input:
            text_data = [user_input]  # Convert input to list
            # Preprocess the new data
            X_new = preprocess_new_data(vectorizer, chi2_features, text_data)
            
            # Make predictions
            y_pred = model.predict(csr_matrix.toarray(X_new))
            
            # Display predictions
            st.write("Predictions:")
            st.write(f"Prediction: {y_pred[0]}")
        else:
            st.error("Please enter some text.")

if __name__ == "__main__":
    main()
