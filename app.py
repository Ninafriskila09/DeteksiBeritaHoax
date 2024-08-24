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
    confm = confusion_matrix(y_test, y_pred, labels=columns)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)
    
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure that y_test and y_pred have the same length
assert len(y_test) == len(y_pred), "y_test and y_pred must be of the same length"

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a DataFrame for y_test and y_pred to compare them side-by-side
comparison_df = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred
})
print("\nComparison of y_test and y_pred:\n", comparison_df.head())

# Compute confusion matrix
confm = confusion_matrix(y_test, y_pred)

# Define column names based on the labels in your classification task
columns = ['0', '1']
df_cm = pd.DataFrame(confm, index=columns, columns=columns)

# Print confusion matrix DataFrame
print("\nConfusion Matrix:\n", df_cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', xticklabels=columns, yticklabels=columns)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


        
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
