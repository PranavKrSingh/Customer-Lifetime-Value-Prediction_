
# 🧠 Customer Lifetime Value Prediction

This project predicts Customer Lifetime Value (CLV) based on historical e-commerce transactions using machine learning models. It also integrates clustering, EDA, and a Streamlit web app for interaction and visualization.



## 📌 Table of Contents


- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Run Streamlit App](#run-streamlit-app)
- [License](#license)


## ✨ Features

- Predict Customer Lifetime Value using ML models (e.g., XGBoost)
- Perform customer segmentation using clustering
- Clean and engineer features from raw e-commerce data
- Visualize key metrics via interactive Streamlit interface
- Retrieve FAQs using LLM integration (optional)

---

## 📁 Project Structure

```

Customer-Lifetime-Value-Prediction\_/
│
├── clustering/               # Customer segmentation logic
│   └── segment.py
│
├── data/                     # Input data and processed files
│
├── eda/                      # Exploratory data analysis notebooks
│
├── feature\_engineering/      # Data cleaning and feature creation
│   └── preprocess.py
│
├── llm/                      # LLM-based FAQ retrieval (optional)
│   └── generate\_response.py
│
├── models/                   # Model training and prediction
│   ├── train\_clv.py
│   ├── predict.py
│   └── clv\_model.pkl         # Saved model
│
├── retriever/                # Vector-based document retriever
│   ├── create\_index.py
│   └── retriever.py
│
├── sql/                      # Optional SQL database setup
│   └── setup\_db.py
│
├── streamlit\_app.py          # Streamlit interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore

````

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PranavKrSingh/Customer-Lifetime-Value-Prediction_.git
cd Customer-Lifetime-Value-Prediction_
````

### 2. Create & activate virtual environment

```bash
python -m venv clv-env
clv-env\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 How It Works

1. **Data Preprocessing**
   → Clean missing values, create RFM (Recency, Frequency, Monetary) features

2. **Model Training**
   → Use `XGBoost` or similar regression model to predict CLV
   → Save model to `models/clv_model.pkl`

3. **Clustering**
   → Cluster customers for segmentation insights

4. **Streamlit App**
   → Input customer features
   → Predict CLV and show segmentation insights

---

## ▶️ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📦 Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn**
* **XGBoost**
* **Joblib**
* **Streamlit**
* **Sentence-Transformers (optional LLM)**
* **FAISS (optional vector store)**

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Pranav Kumar Singh**
🔗 [LinkedIn](https://www.linkedin.com/in/pranavkrsingh)
📫 Email: [pranavkumarsingh32@gmail.com](mailto:pranavkumarsingh32@gmail.com)

