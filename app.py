import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pickled TfidfVectorizer and model
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('mod.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for token in tokens:
        if token.isalnum() and token not in stopwords.words('english') and token not in string.punctuation:
            y.append(ps.stem(token))

    return " ".join(y)

st.title("Email / SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('predict'):
    transformed_sms = transform_text(input_sms)
    print(transformed_sms)
    # Vectorize the preprocessed text
    vectors = tfidf.transform([transformed_sms])
    print(vectors)
    result = model.predict(vectors)[0]
    # Display the result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')



