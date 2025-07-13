import joblib
import os
import pandas as pd
import streamlit as st

from models.predict import predict_single
from retriever.retriever import retrieve
from llm.generate_response import gen_answer

# Load the trained model
model = joblib.load(os.path.join('models', 'clv_model.pkl'))

st.set_page_config(page_title='CLV Predictor', layout='wide')

st.title('🛍️ Customer Lifetime Value Predictor + Q&A')

tab1, tab2, tab3 = st.tabs(['Predict CLV','Customer Segments','Ask the Data'])

with tab1:
    st.header('Predict Individual Customer CLV')

    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input('Last visit (days)', min_value=0)
        frequency = st.number_input('Frequency (#orders)', min_value=0)
    with col2:
        monetary = st.number_input('Total Spent (£)', min_value=0.0)
       # avg_order_value = st.number_input('Average Order Value (£)', min_value=0.0)

    if st.button('Predict'):
         if frequency > 0:
             avg_order_value = monetary / frequency
         else:
             avg_order_value = 0.0
         feats = {
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'AvgOrderValue': avg_order_value
        }

         input_df = pd.DataFrame([feats])
         clv_gbp = model.predict(input_df)[0]

         GBP_TO_INR = 116.0
         clv_inr = clv_gbp * GBP_TO_INR

         st.info(f"Average Order Value auto‑calculated: £{avg_order_value:,.2f}")

         st.success(
             
            f"💰 Predicted Customer Lifetime Value:\n"
            f"   • £{clv_gbp:,.2f}\n"
            f"   • ₹{clv_inr:,.0f}"
        )

# ❌ These three blocks cause an error if the "Predict" button was not clicked
# because `clv_gbp` and `clv_inr` are not defined yet
# ➤ So, either remove them OR wrap them in the same if condition
# For now, remove the extra `st.success()` blocks:

# st.success(
#     f"💰 Predicted Customer Lifetime Value:\n"
#     f"   • £{clv_gbp:,.2f}\n"
#     f"   • ₹{clv_inr:,.0f}"
# )

# st.success(
#     f"💰 Predicted Customer Lifetime Value:\n"
#     f"   • £{clv_gbp:,.2f}\n"
#     f"   • ₹{clv_inr:,.0f}"
# )

with tab2:
    st.header('Customer Segmentation Overview')
    if os.path.exists('data/customer_segments.csv'):
        seg_df = pd.read_csv('data/customer_segments.csv')
        st.dataframe(seg_df.head())
        st.bar_chart(seg_df['Segment'].value_counts())
    else:
        st.warning('Run clustering/segment.py first.')

with tab3:
    st.header('Ask the Data (RAG Q&A)')
    q = st.text_input('Type a question about transactions:')
    if st.button('Ask') and q:
        ctx = '\n'.join(retrieve(q, k=5))
        ans = gen_answer(ctx, q)
        st.write(ans)
