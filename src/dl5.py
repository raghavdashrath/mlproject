import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, padding='post', maxlen=500)  # Padding to 500 words
x_test = pad_sequences(x_test, padding='post', maxlen=500)

model = Sequential([
            Embedding(input_dim=10000, output_dim=128), 
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')  
        ]) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))


score, accuracy = model.evaluate(x_test, y_test, batch_size=64)
print(f"Test accuracy: {accuracy:.4f}")


new_review = "I love this movie, it was amazing!"
new_review_tokens = [imdb.get_word_index().get(word, 0) for word in new_review.split()]  # Tokenizing the new review
new_review_padded = pad_sequences([new_review_tokens], maxlen=500)  # Padding to match the model input size


prediction = model.predict(new_review_padded)
sentiment = "Positive" if prediction >= 0.5 else "Negative"  

print(f"Predicted sentiment: {sentiment}")
