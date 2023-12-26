import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_txt(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  a=[]
  for i in y:
    if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
      a.append(i)
  res=[]
  for i in a:
    res.append(ps.stem(i))
  return " ".join(res);



tfidf=pickle.load(open('vectorizer (1).pkl','rb'))
ml=pickle.load(open('mdl (2).pkl','rb'))
st.title('sms spam :blue[classifier]:sunglasses:')
txt = st.text_area("enter the message")

#preprocess
#vectorization
#model prediction
#display


if st.button('Predict'):
  tx = transform_txt(txt)
    # 2. vectorize
  vect = tfidf.transform([tx])
    # 3. predict
  prd = ml.predict(vect)[0]
  if prd == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")



