import streamlit as st
from transformers import pipeline
import textwrap

st.set_page_config(page_title="Text Summerizer", layout="wide")

st.title("Abstractive text summarizer")

#load model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

#input box
input_text = st.text_area("Enter the text you want to summarize:", height=300)

if st.button("Summarize"):
    if input_text:
        with st.spinner("Summarizing..."):
            summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
            st.subheader("Summary:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summerize.")
