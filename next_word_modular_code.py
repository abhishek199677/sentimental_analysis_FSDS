import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# 1. Prepare your corpus
corpus = [
    "The sunrise over the mountains painted the sky with brilliant shades of orange and pink.",
    "Machine learning algorithms are transforming industries by enabling smarter decision making.",
    "Reading books regularly can improve vocabulary and critical thinking skills.",
    "The city park is a popular place for families to enjoy picnics and outdoor games.",
    "Renewable energy sources like wind and solar are essential for a sustainable future.",
    "Cooking new recipes at home is a fun way to explore different cultures and flavors.",
    "The art museum features exhibitions from both local and international artists.",
    "Learning to play a musical instrument enhances creativity and patience.",
    "Astronauts train for years before embarking on missions to the International Space Station.",
    "Good communication is key to building strong relationships in both personal and professional life.",
    "Gardening is a relaxing hobby that can also provide fresh fruits and vegetables.",
    "Online courses make it possible for people to learn new skills from anywhere in the world.",
    "Practicing mindfulness meditation helps reduce stress and increase focus.",
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
print(predict_next_word("Machine learning", num_words=5))
