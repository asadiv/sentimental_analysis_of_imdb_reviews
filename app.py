import streamlit as st
import joblib
import re

# this was used as a param during training
def tokenizer(text):
    return text.split()

# this function is specifically for the imdb reviews as they contain html tags and emoticons
def preprocess(text):
    # remove html tags
    text = re.sub(r'<[^>]*>', '', text)

    # extract emoticons
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    # clean text
    text = re.sub(r'[\W]+', ' ', text.lower())

    # add emoticons back
    text += ' ' + ' '.join(emoticons).replace('-', '')

    return text

# the data cleaning and all the preprocessing steps + training and expermenting was done in the notebook given
model = joblib.load("sentiment_model1.pkl")

st.title("Movie Review Sentiment Analyzer")

review = st.text_area("Enter a movie review")

if st.button("Predict"):

    processed_review = preprocess(review)

    pred = model.predict([processed_review])[0]
    

    if pred == 1:
        st.success("Positive Review")
    else:
        st.error("Negative Review")