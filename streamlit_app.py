import streamlit as st 
import pandas as pd 
import re
import os
import numpy as np
import yaml
import pickle

## Testing branch


# Set the app title 
st.title('Assignment 2') 
# Add a welcome message 
st.subheader(
    'Image search engine')


def vectorise_document(docs, vocab):
    vectors = np.zeros(len(vocab))
    for token in docs:
        if token in vocab:
            vectors[vocab.index(token)] += 1
    return vectors

def calculate_cosine_similarity(u, v):
    score = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    if np.isnan(score):
      score = 0
    return score

def search_query(input_query, vectors, vocab, data, stop_words):
    pattern = r'\b[A-Za-z]+\b'
    query_words = re.findall(pattern, input_query)
    # lowercase 
    query_words = [word.lower() for word in query_words]
    # remove stopwords
    query_words = [word for word in query_words if word not in stop_words]
    query_vec = vectorise_document(query_words, vocab)
    similarities = [calculate_cosine_similarity(query_vec, vector) for vector in vectors]
    ranking = data.copy()
    ranking['vsm_score'] = similarities
    ranking = ranking.sort_values(by='vsm_score', ascending=False)
    ranking['ranks'] = [i for i in range(1, len(similarities) + 1 )]
    ranking = ranking.loc[ranking['vsm_score'] >= 0.1].iloc[:20]
    ranking_dict = ranking[['image_id','image_name','vsm_score','image_source','photographer']].set_index('image_id').to_dict()
    return ranking_dict
    
st.subheader(
    '**👈 Please select the options on the sidebar to see different analysis performed on the data.**')
genre = st.sidebar.radio(
    "Please select type",
    ('Image Search', 'Seach Examples'))

if genre == 'Image Search':

    widgetuser_input = st.text_input('Enter a search query') 
    
    indexed_images_data = pd.read_csv(r'./data/data.csv')
    
    with open(r'./data/stopwords.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    stopwords = set(config['stop words'])
    
    with open(r'./data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    

    with open(r'./data/vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    ranking_dict = search_query(widgetuser_input, vectors, vocabulary, indexed_images_data, stopwords)
    
    for key,value in ranking_dict['image_name'].items():
        st.write(f"Image ID: {key} | Image Name: {value} | Similarity Score: {ranking_dict['vsm_score'][key]} | Photographer: {ranking_dict['photographer'][key]}")
        
        st.image(ranking_dict['image_source'][key], caption="searched image")
    # st.write('Customized Message:', widgetuser_input)


if genre == 'Seach Examples':
    widgetuser_input = st.text_input('Enter a search query', 'Hello, Streamlit!') 

    st.write('Customized Message:', widgetuser_input)