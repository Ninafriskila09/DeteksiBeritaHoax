import streamlit as st
import pandas as pd
import pickle

with open ('hoax_pickle', 'rb') as file:
    classifier1 = pickle.load(file)
  
# Input teks untuk diprediksi
st.write("Masukkan teks untuk diprediksi:")
input_text = st.text_input("Teks", "")
        if input_text:
            input_text_tfidf = vectorizer.transform([input_text])
            input_text_chi2 = chi2_features.transform(input_text_tfidf)
            input_text_chi2_dense = csr_matrix.toarray(input_text_chi2)
            prediction = model.predict(input_text_chi2_dense)
            sentiment = "Fakta" if prediction[0] == 0 else "Hoax"
            st.write("Hasil prediksi:", sentiment)
    
       

        # Tampilkan Word Cloud
        st.write("Word Cloud untuk Semua Data:")
        all_text = ' '.join(data['clean_text'])
        wordcloud_all = WordCloud(
            width=800, height=400, background_color='white').generate(all_text)
        st.image(wordcloud_all.to_array(), use_column_width=True)

        st.write("Word Cloud untuk Fakta:")
        fakta = data[data['Label'] == 1]
        all_text_fakta = ' '.join(fakta['clean_text'])
        wordcloud_fakta = WordCloud(
            width=800, height=400, background_color='white').generate(all_text_fakta)
        st.image(wordcloud_fakta.to_array(), use_column_width=True)

        st.write("Word Cloud untuk Hoax:")
        hoax = data[data['Label'] == 0]
        all_text_hoax = ' '.join(hoax['clean_text'])
        wordcloud_hoax = WordCloud(
            width=800, height=400, background_color='white').generate(all_text_hoax)
        st.image(wordcloud_hoax.to_array(), use_column_width=True)

        


if __name__ == '__main__':
    main()
