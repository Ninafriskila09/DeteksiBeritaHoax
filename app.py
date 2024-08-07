import streamlit as st
import pandas as pd
import pickle

with open ('hoax_pickle', 'rb') as file:
    hasil = pickle.load(file)
  
# Input teks untuk diprediksi
st.write("Masukkan teks untuk diprediksi:")
input_text = st.text_input("Teks", "")
       

        


if __name__ == '__main__':
    main()
