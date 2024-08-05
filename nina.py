import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Inisialisasi pipeline untuk prediksi (misalnya analisis sentimen)
@st.cache_resource
def load_model():
    model = pipeline('sentiment-analysis')
    return model

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def plot_wordcloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def main():
    st.title("Aplikasi Prediksi dan Wordcloud")

    # Masukkan judul prediksi
    st.header("Masukkan Judul Prediksi")
    input_text = st.text_area("Masukkan teks untuk prediksi:")
    
    if st.button("Deteksi"):
        if input_text:
            # Hasil deteksi
            model = load_model()
            prediction = model(input_text)
            
            st.subheader("Hasil Deteksi")
            st.write(prediction)

            # Tampilan Wordcloud
            wordcloud = generate_wordcloud(input_text)
            wordcloud_img = plot_wordcloud(wordcloud)
            
            st.subheader("Wordcloud")
            st.image(wordcloud_img, use_column_width=True)
        else:
            st.error("Silakan masukkan teks untuk prediksi.")

if __name__ == "__main__":
    main()
