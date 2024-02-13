from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

data = [
    "I love to read books",
    "The quick brown fox jumps",
    "I prefer tea over coffee",
    "Machine learning is an interesting field",
    "Cal Poly Pomona has strong science majors"
]

# Preprocessing the dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data).toarray()

# For a neural network, we need to convert y to categorical data
words = set()
for sentence in data:
    for word in sentence.split():
        words.add(word)

word_to_id = {word: i for i, word in enumerate(words)}
id_to_word = {i: word for word, i in word_to_id.items()}

y = [word_to_id[sentence.split()[-1]] for sentence in data]
y = to_categorical(y, num_classes=len(words))

# Initializing and training the model
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))  # 10 neurons in the first layer
model.add(Dense(len(words), activation='softmax'))  # Output layer with one neuron per unique word

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Improved prediction
# Example prediction
input_string = "Cal Poly Pomona"
prediction_vector = vectorizer.transform([input_string]).toarray()
predicted_probabilities = model.predict(prediction_vector)
predicted_word_id = np.argmax(predicted_probabilities, axis=1)

# Extract the single value from the array to use as a dictionary key
predicted_word_id_value = int(predicted_word_id[0])

if predicted_word_id_value in id_to_word:
    predicted_word = id_to_word[predicted_word_id_value]
    print(f"The predicted next word after '{input_string}' is: {predicted_word}")
else:
    print(f"Predicted ID {predicted_word_id_value} not found in ID to Word Mapping.")
