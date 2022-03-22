from datetime import datetime
from xmlrpc.client import UNSUPPORTED_ENCODING
import streamlit as st
import plotly.express as px 
import altair as alt
import track

import pandas as pd
import numpy as np
import re

import joblib as jb

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

###########################################################
st.set_page_config(layout="wide")
###########################################################

# Models and encoders to be used

def myModel_app_sentiment_ann():
    return load_model('./model/my_neural_network')

def myModel_app_sentiment_lstm():
    return load_model('./model/appSentimentLSTM')

def myModel_sentiment_rnn():
    return load_model('./model/sentimentRNN')

def myModel_sentiment_lstm():
    return load_model('./model/sentimentLSTM')

def myModel_subjective_rnn():
    return load_model('./model/subjectivityRNN')

def myModel_subjective_lstm():
    return load_model('./model/subjectivityLSTM')

def myAppToken():
    return jb.load('apptoken')

def myToken():
    return jb.load('token')

def tfid():
    return jb.load('tfid')

def le_senti():
    return jb.load('label_encoder_sentiment')

def le_sub():
    return jb.load('label_encoder_subjective')

le = WordNetLemmatizer()

def cleanText(txt):
    txt = re.sub(r'@[A-Za-z0-9]+', '', txt)
    txt = re.sub(r'#', '',txt)
    txt = re.sub(r'RT[\s]+','',txt)
    txt = re.sub(r'https?:\/\/\S+','', txt)
    return txt

def text_cleantext(text, stopwords=stopwords.words('english')):
    token = word_tokenize(text.lower())
    word_token = [t for t in token if t.isalpha()]
    clean_token = [t for t in word_token if t not in stopwords]
    lemma_token = [le.lemmatize(t) for t in clean_token]
    return ' '.join(lemma_token)


def predictSentiment(txt,model,token,max_len=255):
    encode_={'joy': 0,'sadness': 1,'fear': 2,'anger': 3,'surprise': 4,'neutral': 5,'disgust': 6,'shame': 7}
    txt = cleanText(txt)
    txt = text_cleantext(txt)
    txt = token.texts_to_sequences([txt])
    txt = sequence.pad_sequences(txt, maxlen=max_len)
    label = model.predict(txt)
    probability = label
    label = np.argmax(label)
    key = list(encode_.keys())
    return key[label], probability


def predictSentimentIntensity(text,model,encoder,token,max_len=180):
    text = cleanText(text)
    text = text_cleantext(text)
    text = token.texts_to_sequences([text])
    text = sequence.pad_sequences(text, maxlen=max_len)
    label = model.predict([text])
    probability = label
    label = encoder.inverse_transform([np.argmax(label,axis=1)])[0]
    return label, probability, encoder

def getPredictions(prediction):
    label, probability, encode_ = prediction
    chart_labels = encode_.classes_
    probability_chart_vals = probability
    df_prob = pd.DataFrame(probability_chart_vals,columns=chart_labels)
    probability = round(probability.max()*100,3)
    proba_df_clean = df_prob.T.reset_index()
    proba_df_clean.columns = ['Intensity','Probability']
    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Intensity',y='Probability',color='Intensity')
    return label,probability,fig

def predictions(text):
    sentiment_model = myModel_app_sentiment_lstm()
    app_token = myAppToken()
    intensity = myModel_sentiment_lstm()
    encoder1 = le_senti()
    my_token = myToken()
    subjective = myModel_subjective_lstm()
    encoder2 = le_sub()

    label, probability = predictSentiment(text,sentiment_model,app_token,)
    encode_={'joy': 0,'sadness': 1,'fear': 2,'anger': 3,'surprise': 4,'neutral': 5,'disgust': 6,'shame': 7}
    chart_labels = encode_.keys()
    probability_chart_vals = probability
    df_prob = pd.DataFrame(probability_chart_vals,columns=chart_labels)
    probability = round(probability.max()*100,3)
    proba_df_clean = df_prob.T.reset_index()
    proba_df_clean.columns = ['Sentiment','Probability']
    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Sentiment',y='Probability',color='Sentiment')
    label1,probability1,fig1 = getPredictions(predictSentimentIntensity(text,intensity,encoder1,my_token))
    label2,probability2,fig2 = getPredictions(predictSentimentIntensity(text,subjective,encoder2,my_token))
    return label,label1,label2

emotions_emoji = {
	"anger":"😠",
	"disgust":"🤮", 
	"fear":"😨😱", 
	"happy":"🤗", 
	"joy":"😂", 
	"neutral":"😐", 
	"sad":"😔", 
	"sadness":"😔", 
	"shame":"😳", 
	"surprise":"😮"}

def main():
    st.title('Sentiment Analyzer')
    menu = ['Home','Moniter','About']
    choice = st.sidebar.selectbox('Menu',menu)    

    if choice == 'Home':
        st.subheader('Home - Sentiment Analyzer')
        track.page_visited_details('Home',datetime.now())
        with st.form(key='emotion_form'):
            text = st.text_area('Type Here')
            submit = st.form_submit_button(label='Submit')

        if submit:

############################################################################
            sentiment_model = myModel_app_sentiment_lstm()
            app_token = myAppToken()
            intensity = myModel_sentiment_lstm()
            encoder1 = le_senti()
            my_token = myToken()
            subjective = myModel_subjective_lstm()
            encoder2 = le_sub()
############################################################################

            label, probability = predictSentiment(text,sentiment_model,app_token,)
            encode_={'joy': 0,'sadness': 1,'fear': 2,'anger': 3,'surprise': 4,'neutral': 5,'disgust': 6,'shame': 7}
            chart_labels = encode_.keys()
            probability_chart_vals = probability
            df_prob = pd.DataFrame(probability_chart_vals,columns=chart_labels)
            probability = round(probability.max()*100,3)
            proba_df_clean = df_prob.T.reset_index()
            proba_df_clean.columns = ['Sentiment','Probability']
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Sentiment',y='Probability',color='Sentiment')
            label1,probability1,fig1 = getPredictions(predictSentimentIntensity(text,intensity,encoder1,my_token))
            label2,probability2,fig2 = getPredictions(predictSentimentIntensity(text,subjective,encoder2,my_token))
            track.track_prediction(text,label,probability,label1,probability1,label2,probability2,datetime.now())

            col1,col2 = st.columns(2)
            with col1:
                st.success('Original Text')
                st.write(text)
                st.success("Sentiment Prediction")
                st.write(label1.title(),'-',label2.title(),'-',label.title(),emotions_emoji[label])
                st.write(f'Confidence : {probability}')
                st.success('Subjective')

            with col2:
                st.success('Probability')
                st.write(probability)
                st.altair_chart(fig,use_container_width=True)


            with col2:
                st.success('Sentiment Intensity Probability')           
                st.write(probability1)
                st.altair_chart(fig1,use_container_width=True)
            

            with col2:
                st.success('Subjectivity Intensity Probability')           
                st.write(probability2)
                st.altair_chart(fig2,use_container_width=True)

    if choice == 'Moniter':
        st.subheader('Monitor App')
        track.page_visited_details('Monitor',datetime.now())
        with st.expander("Page Metrics"):
            page_visited_details = track.all_pages_visited()
            st.dataframe(page_visited_details)	

            pg_count = page_visited_details['name'].value_counts().rename_axis('name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='name',y='Counts',color='name')
            st.altair_chart(c,use_container_width=True)	

            p = px.pie(pg_count,values='Counts',names='name')
            st.plotly_chart(p,use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df = track.all_predictions()
            df = df.astype(str)
            st.dataframe(df)

            prediction_count = df['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='sentiment',y='Counts',color='sentiment')
            st.altair_chart(pc,use_container_width=True)	


    if choice == 'About':
        st.subheader('About')
        track.page_visited_details('About',datetime.now())


if __name__ == '__main__':
    main()