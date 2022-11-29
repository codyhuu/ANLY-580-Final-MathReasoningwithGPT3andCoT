import streamlit as st
import time
import json
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from auto_cot_utils import *

"""
# Math Reasoning & Inference - By GPT-3 and Auto-Chain of Thoughts

🔧 Made by Zheyuan Hu, Muwen You, Ruobing Yan, Dian Zhi

📝 Solve math problems, e.g. jaiocjofampko
"""

if 'options' not in st.session_state:
    st.session_state['options'] = ""

def sidebar_callback():
    st.session_state['options'] = st.session_state['sidebar']
def button1_callback():
    st.session_state['options'] = "foo"
def button2_callback():
    st.session_state['options'] = "bar"

@st.cache
def init_gpt3_qa_system(gpt3_engine):

    # load train question embeddings
    train_q_embeddings = np.load('train_q_embeddings.npy')

    # select 3 as the number of k-means clusters
    K = 3 

    # load train question embedding centroids
    cluster_centers = np.load('train_q_cluster_centers.npy')

    # load train / test data
    train_data = read_jsonl('train.jsonl')
    test_data = read_jsonl('test.jsonl')

    # find the most representative training questions indices
    most_repr_indices = {}
    for c in range(K):
        most_repr_indices[c] = get_nearest_embeddings_idx(cluster_centers[c, :], train_q_embeddings)
    
    # generate auto COT of the most representative questions
    repr_auto_cot = {}
    for key, value in tqdm(most_repr_indices.items()):
        q = get_qa_by_idx(value, train_data)[0]
        prompt = format_prompt(q, keywords=True)
        msg, completion = get_gpt3_response_text(prompt, engine=gpt3_engine)
        repr_auto_cot[key] = prompt + completion + '\n\n'

    # initialize the pretrained sentence bert model
    sbert = SentenceTransformer('all-mpnet-base-v2')

    # initialize the gpt3 qa system
    model = GPT3ArithmeticReasoning(gpt3_engine, sbert, repr_auto_cot, most_repr_indices, train_data, train_q_embeddings, cluster_centers)

    return model, test_data

col1, col2 = st.columns(2)
with col1:
    engine = st.radio('Engine',('text-curie-001', 'text-davinci-003'))
with col2:
    zs = st.checkbox('0-shot')
    zsk = st.checkbox('0-shot w/ keywords')
    acr = st.checkbox('Auto-COT most repr question')
    acn = st.checkbox('Auto-COT nearest question')
    mcr = st.checkbox('Manual-COT most repr question')
    mcn = st.checkbox('Manual-COT nearest question')

api_key = st.text_input('Enter your OpenAI token here', type='password')

input_question = st.text_area('Your math reasoning question here', key= 'options')

def run_callback():
    with st.spinner('Working...'):
        set_openai_apikey(api_key)
        model, test_data = init_gpt3_qa_system(engine)
        st.session_state['options'] = model.generate_gpt3_prompt(input_question, prompt_method = '0-shot')

def button1_callback():
    st.session_state['options'] = "q1 here"
def button2_callback():
    st.session_state['options'] = "q2 here"

st.button('Run', on_click = run_callback)
st.button('Example Question 1', on_click = button1_callback)
st.button('Example Question 2', on_click = button2_callback)

