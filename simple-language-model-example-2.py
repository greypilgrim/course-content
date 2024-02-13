from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Expanded dataset with additional similar questions for the hobby context
data = [
    ("What is your favorite hobby?", "Reading books"),
    ("What do you enjoy doing in leisure?", "Reading novels"),
    ("How do you like to spend your free time?", "Reading"),
    ("What's your go-to activity for relaxation?", "Reading books"),
    ("What animal is known for jumping?", "A fox"),
    ("Which do you prefer, tea or coffee?", "Tea"),
    ("What field studies artificial intelligence?", "Machine learning"),
    ("What university is known for science?", "Cal Poly Pomona")
]

# Preprocess the data
questions = [item[0] for item in data]
answers = [item[1] for item in data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)
y = answers  # The answers are the target variable

# Train the SVM model
model = svm.SVC()
model.fit(X, y)

# Example prediction
input_question = "What do you like to do in your free time?"
prediction_vector = vectorizer.transform([input_question])
predicted_answer = model.predict(prediction_vector)

print(f"The predicted answer to the question '{input_question}' is: {predicted_answer[0]}")
