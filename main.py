import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import tensorflow as tf

#load the imdb dataset
word_index=imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}

#load the pre trained model with relu activation 
model=load_model("D:\krishnaik course projects\SimpleRNN/trained_max_30k.h5")

#helper function to decode the value 
def decode_review(decode_review):
    return ' '.join([revers_word_index.get(i-3,'?') for i in decode_review])
def preprocess(text):
    words=text.lower().split()
    decoded=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([decoded],maxlen=500)
    return padded_review

#prediction function 
def predict_sentiment(review):
    preprocessed_input=preprocess(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


import streamlit as st
# create the streamlit app
st.title("IMDB review Sentiment Analysis")
st.write("Enter a movie to clarify whether it  is positive or negative ")

# user input 
user_input =st.text_area("Movie review")

if st.button('classify'):
    preprocessed_input=preprocess(user_input)
    prediction=model.predict(preprocessed_input) 
    sentiment='positive' if prediction[0][0] >0.5 else 'nagative'

    #display the result
    st.write(f'sentiment : {sentiment}')
    st.write(f'prediction score: {prediction[0][0]}')
else:
    st.write("please enter a movie review you want ")
    