import streamlit as st
import time
import json
import openai
from sentence_transformers import SentenceTransformer

"""
# Math Reasoning & Inference - By GPT-3 and Auto-Chain of Thoughts

🔧 Made by Zheyuan Hu, Muwen You, Ruobing Yan, Dian Zhi

📝 Solve math problems, e.g. jaiocjofampko
"""


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        engin = st.radio('Engine',("curie", "davinci"))
    with col2:
        method = st.radio('Method', ('Zero-shot', 'Few-shot', 'Few-shot with AUTO COT'))

with st.container():
    access = st.text_area('You may need to add your OpenAI token here, if ours runs out.')

with st.container():
    inputs = st.text_area('Your math reasoning question here', value = "Q: vhacuodjendan ofamcoemkanmvoea\nA: ")
    if st.button('Run'):
        with st.spinner('Working...'):
            # outputs = func(inputs, **arg)
            time.sleep(1) ## replace this
        st.write('outputs')

model = SentenceTransformer('all-mpnet-base-v2')

test = model.encode('a, b, c')

st.write(str(test))




