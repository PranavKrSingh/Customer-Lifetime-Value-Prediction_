import pandas as pd, numpy as np, os, faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join('data', 'clean_transactions.csv')
INDEX_PATH = os.path.join('retriever', 'faiss.index')
TEXT_PATH  = os.path.join('retriever', 'index_texts.txt')
MODEL_NAME = 'all-MiniLM-L6-v2'

df = pd.read_csv(DATA_PATH).head(10000)  # limit for demo

# Fixing column names to match your dataset
docs = df.apply(lambda r: f"Invoice {r.Invoice} bought {r.Description} for £{r.TotalPrice} on {r.InvoiceDate} by customer {r['Customer ID']}", axis=1).tolist()

model = SentenceTransformer(MODEL_NAME)
emb = model.encode(docs, show_progress_bar=True).astype('float32')

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)
faiss.write_index(index, INDEX_PATH)

with open(TEXT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(docs))

print('✅ FAISS index built with', len(docs), 'docs')
