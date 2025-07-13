import streamlit as st
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load models and data
MODEL_PATH = 'models/clv_model.pkl'
INDEX_PATH = 'retriever/faiss.index'
TEXT_PATH = 'retriever/index_texts.txt'
MODEL_NAME = 'all-MiniLM-L6-v2'

st.set_page_config(page_title="Customer Lifetime Value & RAG Chatbot", layout="wide")
st.title("🤖 Customer Lifetime Value Prediction + RAG Q&A Chatbot")

# Load ML model
model = joblib.load(MODEL_PATH)

# Load FAISS index and doc texts
index = faiss.read_index(INDEX_PATH)
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    docs = f.readlines()

retriever = SentenceTransformer(MODEL_NAME)

# Tabs
tab1, tab2 = st.tabs(["💡 Ask a Question (RAG)", "📈 Predict CLV"])

with tab1:
    st.subheader("💬 Ask about past customer transactions")

    user_query = st.text_input("Type your question:")
    if user_query:
        query_vec = retriever.encode([user_query]).astype('float32')
        D, I = index.search(query_vec, k=3)
        st.markdown("#### Top Matches:")
        for i in I[0]:
            st.write(f"• {docs[i]}")

with tab2:
    st.subheader("🧮 Enter Customer RFM Values")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, value=10)
    with col2:
        frequency = st.number_input("Frequency (# of invoices)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary Value (total £)", min_value=1.0, value=100.0)

    avg_order_value = monetary / frequency

    if st.button("🚀 Predict CLV"):
        input_df = pd.DataFrame([[recency, frequency, monetary, avg_order_value]],
                                columns=['Recency', 'Frequency', 'Monetary', 'AvgOrderValue'])
        pred = model.predict(input_df)[0]
        st.success(f"💰 Predicted Customer Lifetime Value: £{pred:.2f}")
