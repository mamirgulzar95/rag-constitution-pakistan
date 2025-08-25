import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Load preprocessed data ---
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open("faiss_index.bin", "rb") as f:
    #index = faiss.read_index(faiss.IOReader(f))
    # Load FAISS index correctly
    index = faiss.read_index("faiss_index.bin")


# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# --- Streamlit UI ---
st.title("📜 Constitution of Pakistan - AI Q&A")

user_question = st.text_input("Ask a question about the Constitution:")

if user_question:
    # Step 1: Embed question
    query_embedding = embedder.encode([user_question]).astype("float32")

    # Step 2: Search
    _, indices = index.search(query_embedding, k=3)
    context = "\n".join([chunks[i] for i in indices[0]])

    # Step 3: Generate Answer
    result = qa_pipeline({
        'context': context,
        'question': user_question
    })

    st.markdown("**Answer:**")
    st.success(result['answer'])
