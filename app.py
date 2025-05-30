import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("next_word_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
total_words = len(tokenizer.word_index) + 1

# Find max sequence length (needed for padding)
def get_max_seq_len(tokenizer, corpus):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            ngram_seq = token_list[:i+1]
            input_sequences.append(ngram_seq)
    max_seq_len = max(len(x) for x in input_sequences)
    return max_seq_len

# Use the same corpus as in training
corpus = [
    "Abhishek commitment to affordable education wasn't just a business strategy—it was his life's mission. Over the years, iNeuron has helped over 1.5 million students from 34+ countries, providing them with the skills they need to succeed in today's competitive job market. Many of these students, like Abhishek himself, came from disadvantaged backgrounds. They saw iNeuron as a lifeline—an opportunity to rise above their circumstances.",
    "In 2022, iNeuron was acquired by PhysicsWallah in a deal worth ₹250 crore. While this acquisition was a significant milestone, Abhishek remained focused on his mission. Even after the acquisition, iNeuron continued to offer some of the most affordable and accessible tech courses in the world.",
    "deep learning is a branch of machine learning",
    "natural language processing is a field of AI",
    "AI is the future",
    "I enjoy teaching AI",
    "students love AI projects",
    "learning new things is exciting",
    "learning new things is fun",
    "learning new things is rewarding",
    "learning new things is enjoyable",
    "teaching AI is rewarding",
    "learning new things is enjoyable",
]
max_seq_len = get_max_seq_len(tokenizer, corpus)

# Prediction function
def predict_next_words(seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_word_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_word_index:
                output_word = word
                break
        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

# Streamlit UI
st.title("📝 Next-Word Predictor (LSTM Language Model)")
st.write("Enter a seed text and let the model generate the next words!")

seed_text = st.text_input("Enter your seed text:", "natural language is")
num_words = st.slider("Number of words to predict:", 1, 10, 3)

if st.button("Predict"):
    result = predict_next_words(seed_text, num_words)
    st.success(f"**Generated Text:** {result}")
