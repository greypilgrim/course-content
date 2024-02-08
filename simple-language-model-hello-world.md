### A Simple Language Model Training 

To illustrate how to train a basic language model using scikit-learn (sklearn), let's create a simple example. We will use a dummy text dataset consisting of few sentences. Here's an example dataset:

```python
data = [
    "I love to read books",
    "The quick brown fox jumps",
    "I prefer tea over coffee",
    "Machine learning is an interesting field"
]
```

Now, let's dive into the training process using sklearn. First, we need to import the necessary modules and preprocess the data.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# Preprocessing the dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
y = []

for i in range(len(data)):
    words = data[i].split()  # Split the sentence into words
    next_word = words[-1]    # Select the last word as the target
    y.append(next_word)

# Initializing and training the SVM model
model = SVC()
model.fit(X, y)
```

Here, we use `CountVectorizer` to convert the sentences into numerical vectors. Then, we split the dataset into input features `X`, which represents the sentences, and the target labels `y`, which represent the next word to predict.

Now we can proceed with training the model. To keep this example simple, we are using the `MultinomialNB` classifier from sklearn. However, note that in real-world cases, more advanced models like recurrent neural networks (RNNs) or transformers are often used for language modeling.

Once the model is trained, it can be used to predict the next word based on a given input string. Here's an example prediction:

```python
# Example prediction
input_string = "I prefer"
prediction_vector = vectorizer.transform([input_string])
predicted_word = model.predict(prediction_vector)

print(f"The predicted next word after '{input_string}' is: {predicted_word[0]}")
```

This prediction will output the most likely next word after the input string "I prefer".

Keep in mind that this example is quite basic, and real-world language models often require more complex architectures and training procedures for better performance.

### How the CountVectorizer Works?

`CountVectorizer` is a useful tool in scikit-learn for converting a collection of text documents into a matrix of token counts. It takes in a text dataset, tokenizes the text into individual words, and counts the occurrence of each word in the dataset.

Here's an example to demonstrate how `CountVectorizer` works:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = [
    "I love to read books",
    "The quick brown fox jumps",
    "I prefer tea over coffee",
    "Machine learning is an interesting field"
]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the data
X = vectorizer.fit_transform(data)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names()

# Print the transformed data
print("Transformed Data:")
print(X.toarray())
print()

# Print the feature names
print("Feature Names:")
print(feature_names)
```

In this code, we start by importing `CountVectorizer` from scikit-learn. We then define a sample dataset consisting of a few sentences.

Next, we initialize `CountVectorizer` by creating an instance of the class. We don't have to specify any parameters in this example, but `CountVectorizer` has many options for customizing the tokenization process, such as handling n-grams or excluding stop words.

After initialization, we use `fit_transform()` to preprocess and transform the data into a matrix of token counts. The result is stored in the variable `X`.

To get the feature names (words) corresponding to the columns of the transformed matrix, we use the `get_feature_names_out()` method of `CountVectorizer`.

Finally, we print the transformed data, which is the matrix of token counts represented as a 2D array, and the feature names (words).

When you run this code, you should see output similar to the following:

```
Transformed Data:
[[0 0 1 1 1 0 1 0]
 [1 1 0 0 0 1 0 1]
 [0 0 0 0 0 1 0 0]
 [0 0 0 0 0 1 0 0]]

Feature Names:
['books', 'brown', 'coffee', 'field', 'fox', 'interesting', 'jumps', 'learning', 'love', 'machine', 'over', 'prefer', 'quick', 'read', 'tea', 'the', 'to']
```

The transformed data matrix represents the occurrence of each word from the dataset. Each row corresponds to a sentence, and each column represents a word. The values in the matrix indicate how many times each word appears in each sentence.

Additionally, the feature names list provides the ordered words (features) corresponding to each column.

That's a basic overview of how `CountVectorizer` works. It's a powerful tool for transforming text data into a numerical representation that can be used with machine learning models.

Simple words like "I" is not included in the Vectorizer by deafult. You can still enable to include them in the library settings. 


### Sample Code Reference

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 

data = [
    "I love to read books",
    "The quick brown fox jumps",
    "I prefer tea over coffee",
    "Machine learning is an interesting field",
    "Cal Poly Pomona has strong science majors"
]

# Preprocessing the dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
y = []

for i in range(len(data)):
    words = data[i].split()  # Split the sentence into words
    next_word = words[-1]    # Select the last word as the target
    y.append(next_word)

print("x: ", X)
print("x: ", X.toarray())
print("Features: ", vectorizer.get_feature_names_out())
print("y: ", y)

# Initializing and training the model
model = svm.SVC()
model.fit(X, y)

# Example prediction
input_string = "Cal Poly Pomona"
prediction_vector = vectorizer.transform([input_string])
predicted_word = model.predict(prediction_vector)

print(f"The predicted next word after '{input_string}' is: {predicted_word[0]}")

```

https://replit.com/@andrews0672/GrownParallelAlgorithms#main.py

### Other Text Vectorizers

Besides CountVectorizer, there are several other vectorizers that can be used to train a simple language model. Some of them include:

1. TfidfVectorizer: This vectorizer converts text documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. It assigns weights to each term based on its relevance to the document and the corpus. This vectorizer captures the importance of terms in the documents.

2. HashingVectorizer: This vectorizer applies the hashing trick to tokenize the text documents and convert them into a fixed-size vector representation. It avoids the need to maintain a vocabulary and reduces the memory footprint. However, it does not provide the feature names like CountVectorizer.

3. Word2VecVectorizer: Word2Vec is a popular word embedding technique that represents words as dense vectors in a high-dimensional space. The Word2VecVectorizer assigns a vector representation to each word in the text, capturing the semantic meaning of the words based on their context. It can be used to create powerful word embeddings for language models.

4. GloVeVectorizer: GloVe (Global Vectors for Word Representation) is another word embedding technique that represents words as vectors. Similar to Word2Vec, GloVeVectorizer captures semantic meaning, but it utilizes global word co-occurrence statistics. It creates word embeddings by factorizing a word co-occurrence matrix.

5. BERTVectorizer: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model that has revolutionized natural language processing tasks. BERTVectorizer uses a pre-trained BERT model to encode text into contextualized word embeddings. It considers the contextual meaning of words based on their surrounding words.

These vectorizers provide different approaches to representing text data, and their usage depends on the specific requirements and context of the language model.

### Use TfidfVectorizer

Title: Introduction to TF-IDF Vectorizer

Objective: Understand the concept of TF-IDF vectorizer and its importance in text analysis.

Lesson:

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that aims to reflect the importance of a word in a document within a larger collection of documents. It is commonly used in natural language processing tasks, such as text classification and information retrieval.

TF (Term Frequency) represents the frequency of a word in a document. It measures how often a word appears within a specific document. The higher the count, the more important the word may seem in that document.

IDF (Inverse Document Frequency) measures the significance of a word in a corpus of multiple documents. It helps to identify words that are unique or rarely occurring across the entire corpus.

The TF-IDF Vectorizer combines these two metrics to create a numerical vector representation of a text document. Let's understand its working principle with a simple example.

Example:

Consider a collection of four documents:

Document 1: "I love to read books"  
Document 2: "The quick brown fox jumps"  
Document 3: "I prefer tea over coffee"  
Document 4: "Machine learning is an interesting field"  

Step 1: Calculating Term Frequency (TF)

For each document, we count the frequency of each word. For instance, in Document 1, the word "I" appears once, "love" appears once, "to" appears once, "read" appears once, and "books" appears once.

Step 2: Calculating Inverse Document Frequency (IDF)

We calculate the IDF for each unique word in the corpus. IDF is calculated as the logarithm of the total number of documents divided by the number of documents containing the specific word.

For example, considering the word "the" (which appears in Document 2) - the IDF would be log(4/1) = log(4) = 0.602.

Step 3: TF-IDF Vectorization

Finally, we calculate the TF-IDF score for each word in each document. The score is computed by multiplying the TF and IDF values.

For example, the TF-IDF score for the word "love" in Document 1 would be: TF("love") * IDF("love") = 1 * IDF("love").

Step 4: Vector Representation

The TF-IDF vectorizer assigns a vector representation to each document. Each dimension of the vector corresponds to a unique word in the corpus, and the value in each dimension represents the TF-IDF score of that word in the document.

For example, the vector representation for Document 1 might look like [0, 1.386, 0, 0, 0], where the value 1.386 corresponds to the TF-IDF score of the word "love."

Using the TF-IDF vectorizer, we can convert a collection of textual documents into a matrix representation suitable for training machine learning models.

Benefits of TF-IDF Vectorizer:
1. It captures the importance of words in a document, giving more weight to rare and informative terms.
2. It reduces the impact of commonly occurring words that might not carry much meaning (e.g., "the," "is," "and").
3. It helps to identify key terms and topics within a corpus of documents.

By utilizing TF-IDF vectorization, we can extract meaningful features from text data and enable various natural language processing tasks.

Practical applications include:
- Sentiment analysis
- Text classification
- Information retrieval
- Keyword extraction

With an understanding of TF-IDF vectorization, you will be equipped to work with text data and build more sophisticated language models.

Keep learning and exploring the fascinating world of natural language processing!

### TfidfVectorizer in Action

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm 

data = [
    "I love to read books",
    "The quick brown fox jumps",
    "I prefer tea over coffee",
    "Machine learning is an interesting field",
    "Cal Poly Pomona has strong science majors"
]

# Preprocessing the dataset
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
y = []

for i in range(len(data)):
    words = data[i].split()  # Split the sentence into words
    next_word = words[-1]    # Select the last word as the target
    y.append(next_word)

print("x: ", X)
print("x: ", X.toarray())
print("Features: ", vectorizer.get_feature_names_out())
print("y: ", y)

# Initializing and training the model
model = svm.SVC()
model.fit(X, y)

# Example prediction
input_string = "Cal Poly Pomona"
prediction_vector = vectorizer.transform([input_string])
predicted_word = model.predict(prediction_vector)

print(f"The predicted next word after '{input_string}' is: {predicted_word[0]}")
```



