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
    

        


if __name__ == '__main__':
    main()
