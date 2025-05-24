import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load the pre-trained model and tokenizer
file_path = "/Users/saurabhbiswal/Documents/Udemy/AI_Repo/E2E_LSTM_GRU/models"
model = load_model(os.path.join(file_path, "hamlet_lstm_model.h5"))

tokenizer_path = os.path.join(file_path, "hamlet_tokenizer.pickle")
with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Preprocess the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[
            -(max_sequence_length - 1) :
        ]  # Keep only the last max_sequence_length - 1 tokens
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_length - 1, padding="pre"
    )

    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Get the word from the index
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word


# Streamlit app
st.title("Next Word Prediction with LSTM")
st.write("Enter a sentence to predict the next word:")

# Text input box appears immediately
input_text = st.text_input("Input Text", "To be or not to be")

# Predict only when the button is clicked
if st.button("Predict Next Word"):
    if input_text:
        max_sequence_length = model.input_shape[1] + 1  # Get max length from model
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
        st.write(f"The predicted next word is: **{next_word}**")
    else:
        st.write("Please enter some text to predict the next word.")
