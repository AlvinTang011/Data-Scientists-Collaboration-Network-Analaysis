import os
import streamlit as st
import streamlit.components.v1 as components

st.title(f'Network in year:')
st.sidebar.title('Select year to visualize')
cumulative = st.sidebar.checkbox('Cumulative')
rootdir = './Results/years' if not cumulative else './Results/cumulative_years'
options = sorted(x.split('.')[0] for x in os.listdir(rootdir))
year = st.sidebar.radio('year', options)
st.subheader(year)

file = f'{rootdir}/{year}.html'
with open(file, 'r', encoding='utf-8') as f:
    source_code = f.read()
    components.html(source_code, height = 1200,width=1000)
