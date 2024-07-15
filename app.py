import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

# text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    #include stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    # used stemming using porter stemmer
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
# importing both the pickle files that we used to train model and vectorizer as tfidf and naive bayes
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#title of the app
st.title("Email/SMS Spam Classifier")

#input area
input_sms = st.text_area("Enter the message")

#predict button
if st.button('Predict'):

    # preprocess the sms
    transformed_sms = transform_text(input_sms)
    # vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
