import streamlit as st
from transformers import pipeline
import textwrap
import fitz  # PyMuPDF
import trafilatura
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Set up Streamlit page
st.set_page_config(page_title="📄 Advanced Text Summarizer", layout="wide")
st.title("🧠 Text Summarizer with PDF & URL Support")

# Load summarizer model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Function to chunk large text
def split_into_chunks(text, max_token_len=1000):
    return textwrap.wrap(text, max_token_len)

# Summarize long input
def summarize_long_text(text):
    chunks = split_into_chunks(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from URL using trafilatura
def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)

# User Interface - Tabs for Input Types
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📁 PDF Upload", "🌐 URL Input"])

# 📝 Text Input Tab
with tab1:
    input_text = st.text_area("Enter long text to summarize:", height=300)
    if st.button("Summarize Text"):
        if input_text.strip():
            with st.spinner("🔄 Summarizing..."):
                lottie_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_j1adxtyb.json")  # example animation
                st_lottie(lottie_animation, height=150, key="summarize")

                summary = summarize_long_text(input_text)
            st.subheader("Summary")
            st.success(summary)
        else:
            st.warning("Please enter some text.")


# 📁 PDF Upload Tab
with tab2:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if st.button("Summarize PDF"):
        if uploaded_file:
            with st.spinner("🔄 Extracting and summarizing your PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                summary = summarize_long_text(pdf_text)
            st.subheader("Summary")
            st.success(summary)
        else:
            st.warning("Please upload a valid PDF.")


# 🌐 URL Input Tab
with tab3:
    url_input = st.text_input("Enter a URL (e.g., blog or article):")
    if st.button("Summarize URL"):
        if url_input.strip():
            with st.spinner("🔄 Summarizing content from the URL..."):
                try:
                    url_text = extract_text_from_url(url_input)
                    if url_text:
                        summary = summarize_long_text(url_text)
                        st.subheader("Summary")
                        st.success(summary)
                    else:
                        st.error("Could not extract clean text from the URL.")
                except Exception as e:
                    st.error(f"Error extracting from URL: {e}")
        else:
            st.warning("Please enter a valid URL.")

