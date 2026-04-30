import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open('pipeline.pkl' , 'rb'))

# df = pickle.load(open(r'C:\Users\Abhishek sharma\Artificial Intelligence\Machine Learning\Projects\IPl Win Probability\dataframe.pkl' , 'rb'))


st.set_page_config(page_title="IPL win predictor -First Inning",page_icon="🧊",initial_sidebar_state="expanded")
st.title("IPL Win Predictor - First Innings")
df = pd.read_csv('matches.csv')

team1 = st.selectbox("Choose First Team" , options=['Choose from options'] + list(df['team1'].unique()))
team2 = st.selectbox("Choose Secoond Team" , options=['Choose from options'] + list(df['team2'].unique()))
tosswinner = st.selectbox("Choose Tos  Winner" , options=['Choose from options'] + list([team1 , team2]))
toss_decision = st.selectbox("Choose Toss Decision" , options=['Choose from options'] + list(df['toss_decision'].unique()))
result = st.selectbox("Choose Result" , options=['Choose from options'] + list(df['result'].unique()))
city = st.selectbox("Choose City Name" , options=["Choose from options"] + list(df['city'].unique()))
venue = st.selectbox("Choose Venue of Match" , options=['Choose from options'] + list(df['venue'].unique()))

data = {
    "city" : [city],
    "team1" : [team1],
    "team2" : [team2],
    "toss_winner" : [tosswinner],
    "toss_decision" : [toss_decision],
    "result" : [result],
    "venue" : [venue]

}

df2 = pd.DataFrame(data)
if st.button("Predict"):
    y_pred = model.predict_proba(df2)
    t1 = y_pred[0][0]
    t2 = y_pred[0][1]
    st.markdown(f"Chances of :green[{team1}] : :red[{np.round(t1, 4)}%]")
    st.markdown(f"Chances of :green[{team2}] : :red[{np.round(t2, 4)}%]")

    y = model.predict(df2)
    if y == 0:
        st.header(f"High Chances for :red[{team1}]")
    else:
        st.header(f"High Chances for :red[{team2}]")

