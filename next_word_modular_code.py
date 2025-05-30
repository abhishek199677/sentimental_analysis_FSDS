import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# 1. Prepare your corpus
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

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# 3. Create input sequences using n-grams
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        ngram_seq = token_list[:i+1]
        input_sequences.append(ngram_seq)

# 4. Pad sequences
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# 5. Split into predictors and label
x = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 6. Build the model
model = Sequential([
    Embedding(total_words, 32, input_length=max_seq_len-1),
    LSTM(64),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Train the model
model.fit(x, y, epochs=400, verbose=1)

# 8. Save the model
model.save("next_word_model.h5")
print("Model saved as next_word_model.h5")

# 9. Save the tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as tokenizer.pkl")

# 10. (Optional) Function to predict next word(s)
def predict_next_word(seed_text, num_words=5):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_word_index = np.argmax(predicted)
        for word, index in tokenizer.word_index.items():
            if index == next_word_index:
                seed_text += ' ' + word
                break
    return seed_text

# Example usage
print(predict_next_word("natural language is", num_words=5))
