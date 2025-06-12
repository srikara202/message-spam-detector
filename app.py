import streamlit as st
import pickle
import nltk

@st.cache_resource
def download_nltk_resources():
    # download only once per deployment (or until code changes)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return True

# trigger the download
download_nltk_resources()

from nltk.corpus import stopwords
import sklearn
import string
from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

tfidf = pickle.load(open( "vectorizer.pkl", "rb" ))
model = pickle.load(open( "model.pkl", "rb" ))

st.title("Spam Detector")

input_sms = st.text_area("Enter the message (email or SMS)")

def transform_text(text):
    text = text.lower()  # conversion to lower case
    text = nltk.word_tokenize(text)  # tokenization

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")