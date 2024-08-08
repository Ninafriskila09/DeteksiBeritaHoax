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

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title('Text Classification App')

    # Muat objek yang telah disimpan
    vectorizer, chi2_features, model = load_objects('vectorizer.pkl', 'chi2_features.pkl', 'model.pkl')

    st.sidebar.header('Upload Your Data')
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['xlsx'])
    
    if uploaded_file:
        data = pd.read_excel(uploaded_file)
        if 'clean_text' not in data.columns:
            st.error("File must contain a 'clean_text' column.")
        else:
            st.write("Data Sample:")
            st.write(data.head())

            text_data = data["clean_text"]

            # Preprocess the new data
            X_new = preprocess_new_data(vectorizer, chi2_features, text_data)

            # Make predictions
            y_pred = model.predict(csr_matrix.toarray(X_new))
            
            # Display predictions
            data['Predictions'] = y_pred
            st.write("Predictions:")
            st.write(data[['clean_text', 'Predictions']])
            
            # Optionally, if you have true labels in the uploaded file:
            if 'Label' in data.columns:
                y_true = data["Label"]
                st.write("Classification Report:")
                st.text(classification_report(y_true, y_pred))

                columns = sorted(y_true.unique())
                confm = confusion_matrix(y_true, y_pred, labels=columns)
                df_cm = pd.DataFrame(confm, index=columns, columns=columns)

                st.write("Confusion Matrix:")
                st.write(df_cm)

if __name__ == "__main__":
    main()
