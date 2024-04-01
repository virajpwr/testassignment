import streamlit as st 
import pandas as pd 
# Set the app title 
st.title('Assignment 2') 
# Add a welcome message 
st.write('Welcome to my Streamlit app!') 
# Create a text input 
widgetuser_input = st.text_input('Enter a search query', 'Hello, Streamlit!') 

st.write('Customized Message:', widgetuser_input)

df = pd.read_csv(r'./data/data.csv')

st.write(df)
st.image('https://cdn.openphoto.net/thumbs2/volumes/mike/20171103/openphotonet_MAJ_4498.JPG', caption="searched image")