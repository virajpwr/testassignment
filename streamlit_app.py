import streamlit as st 
import pandas as pd 
import numpy as np
import yaml
import pickle


from tqdm import tqdm
 
# utilities
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from copy import deepcopy
import os
from itertools import chain
import operator


import math
from functools import lru_cache
import math
import operator
from tqdm import tqdm
import itertools
from collections import Counter
from dataclasses import dataclass
from math import log
import pickle

from src import preprocess, invertedindex, model



def mapping(query, doc):    
    ''' 
    - Map each query id with all the document ids.
    
    :param query: dictionary with queryid and query as key-value pair.
    :param doc: dictionary with docid and text as key-value pairs.
    
    :return: dictionary with queryid as key and list of document ids as value.
    '''    
    mappings = {}
    for key, value in query.items():
        for k, v in doc.items():
            doc_keys = list(doc.keys())
            doc_keys = [int(i) for i in doc_keys]
            mappings[int(key)] = doc_keys 
    return mappings

def generate_index(doc_collection):
    index = invertedindex.InvertedIndexDocs(doc_collection)
    index.create_index()
    return index


# Set the app title 
st.title('Assignment 2') 

st.subheader(
    '**👈 Please select the options on the sidebar for search query examples**')

# Add a welcome message 
st.subheader(
    'Image search engine')


def search_query(input_query, docs, inv):
    map_doc_queries = mapping(input_query, docs)
    model = model.VSM(inv, map_doc_queries)

    # loop through each query and perform ranking
    ranking = {qid: model.rank(qid, query) for qid, query in clean_query.items()}

    filter_rank = {k: v for k, v in ranking[1].items() if v > 0.1}
    result = defaultdict(dict)
    for key, value in filter_rank.items():
        result[key]['score'] = value
        result[key]['url'] = urls[key]
        result[key]['image_name'] = image_data[key]['image_name']
        result[key]['photographer'] = image_data[key]['photographer']
        result[key]['image_id'] = key
    return ranking_dict
    

genre = st.sidebar.radio(
    "Please select type",
    ('Image Search', 'Seach Examples'))

if genre == 'Image Search':
    with open(r'./data/image_data.pkl', 'rb') as f:
        image_data = pickle.load(f)
        

    urls= {image_id: image_data[image_id]['image_source'] for image_id in image_data}
    documents= {image_id: image_data[image_id]['image_details'] for image_id in image_data}
    
    inv_index = generate_index(documents)
    
    with open(r'./data/stopwords.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    stopwords = set(config['stop words'])

    widgetuser_input = st.text_input('Enter a search query') 
    
    pattern = r'\b[A-Za-z]+\b'
    query_words = re.findall(pattern, widgetuser_input)

    # lowercase 
    query_words = [word.lower() for word in query_words]
    # remove stopwords
    query_words = [word for word in query_words if word not in stopwords]

    query = {1:query_words}
    clean_query = {key: ' '.join(value) for key, value in query.items()}

    ranking_dict = search_query(widgetuser_input, documents, inv_index)
    
    for key, value in ranking_dict.items():
        st.write(f'Image ID {value["image_id"]} | Image Name {value["image_name"]} | Similarity Score: {value["score"]} | Photographer: {value["photographer"]}') 
        st.image(value["url"], caption="searched image")


if genre == 'Seach Examples':
    widgetuser_input = st.text_input('Enter a search query', 'Hello, Streamlit!') 

    st.write('Customized Message:', widgetuser_input)