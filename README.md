# Customer Lifetime Value (CLV) Prediction – End‑to‑End Project

This repository is the **capstone** for an 8‑week Data‑Science‑with‑GenAI internship.  
It demonstrates every skill learned — from Python and SQL through Feature Engineering, ML, Clustering, and Retrieval‑Augmented Generation (RAG).

## 📚 Learning Path ↔️ Project Mapping
| Week | Internship Topic           | Where It Appears Here |
|------|----------------------------|-----------------------|
| 1    | Python basics              | Clean, modular scripts (`*.py`) |
| 2    | OOP                        | `models/clv_model.py` class & wrappers |
| 3    | Data Science               | `eda/eda.ipynb` notebook |
| 4    | Feature Engineering        | `feature_engineering/preprocess.py` |
| 5    | Regression / Prediction    | `models/train_clv.py` (XGBoostRegressor) |
| 6    | Clustering                 | `clustering/segment.py` (K‑means on RFM) |
| 7    | SQL Basics                 | `sql/setup_db.py` + ad‑hoc queries |
| 8    | GenAI (RAG)                | `retriever/` + `llm/` + Streamlit chat tab |

## 🔧 Quick Start
```bash
pip install -r requirements.txt

# 1️⃣ Prepare data
python feature_engineering/preprocess.py --in data/online_retail_II.xlsx

# 2️⃣ Train CLV regressor
python models/train_clv.py

# 3️⃣ Segment customers
python clustering/segment.py

# 4️⃣ Build FAISS index for Q&A
python retriever/create_index.py

# 5️⃣ Load into SQLite (optional)
python sql/setup_db.py

# 6️⃣ Launch Streamlit UI
streamlit run streamlit_app.py
```