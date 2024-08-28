import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from scipy.sparse import csr_matrix

# Load the model and vectorizer
model = joblib.load('model.pkl')  # Assuming 'model.pkl' is your trained model file
vectorizer = joblib.load('vectorizer.pkl')  # Assuming 'vectorizer.pkl' is your vectorizer file

# Load the dataset
dataset = pd.read_excel('dataset_clean.xlsx')

def load_data():
    return dataset

def preprocess_data(data):
    X_raw = data["clean_text"]
    y_raw = data["Label"]
    X_TFIDF = vectorizer.transform(X_raw)
    return X_TFIDF, y_raw

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
    st.write("Word Cloud for All Data:")
    all_text = ' '.join(data['clean_text'])
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), use_column_width=True)

    st.write("Word Cloud for Facts:")
    fakta = data[data['Label'] == 1]
    all_text_fakta = ' '.join(fakta['clean_text'])
    wordcloud_fakta = WordCloud(width=800, height=400, background_color='white').generate(all_text_fakta)
    st.image(wordcloud_fakta.to_array(), use_column_width=True)

    st.write("Word Cloud for Hoaxes:")
    hoax = data[data['Label'] == 0]
    all_text_hoax = ' '.join(hoax['clean_text'])
    wordcloud_hoax = WordCloud(width=800, height=400, background_color='white').generate(all_text_hoax)
    st.image(wordcloud_hoax.to_array(), use_column_width=True)

def load_html():
    html_file_path = "index.html"
    try:
        with open(html_file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"HTML file not found at path: {html_file_path}")
        return ""

def main():
    st.title("HTML Viewer and Hoax Detection")

    # Load HTML content
    html_content = load_html()
    
    # Display HTML content if available
    if html_content:
        st.markdown(html_content, unsafe_allow_html=True)

    # Load data and preprocess
    data = load_data()
    X_features, y_labels = preprocess_data(data)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Input text for prediction
    st.markdown("**Enter Text for Prediction**")
    input_text = st.text_area("", height=150)

    # Detection button
    detect_button = st.button("Detect")

    # Display results
    if detect_button and input_text:
        # Transform text with vectorizer used to train the model
        input_text_tfidf = vectorizer.transform([input_text])
        input_text_dense = csr_matrix.toarray(input_text_tfidf)

        # Predict using the model
        prediction = model.predict(input_text_dense)
        sentiment = "Fact" if prediction[0] == 0 else "Hoax"

        # Display result
        st.markdown(f"**{sentiment}**")

        # Evaluate the model
        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_evaluation(y_test, y_pred)

        # Display Word Clouds
        display_wordclouds(data)

if __name__ == '__main__':
    main()

