import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    dt = pickle.load(f)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    clean_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words and len(word) > 2]
    
    # Join the words back into a single string
    return ' '.join(clean_words)

def predict_spam(text):
    clean = clean_text(text)
    vector = tfidf.transform([clean])
    pred = dt.predict(vector)
    return pred[0]

st.title('Spam Classifier')

text_input = st.text_area("Enter text")

if st.button('Predict'):
    if text_input:
        pred = predict_spam(text_input)
        st.write("Prediction:", pred)

st.markdown("""
**Note:** This model uses TF-IDF, so it gives better results on the "spam.csv" data. But it doesn't understand semantic relationships,
so if you provide data from a different source, it will not give good results.
For a more flexible model, Word2Vec is better because it understands semantic relationships between two words,
but it will increase computational time and size.
""")
