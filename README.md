# Customer-Lifetime-Value-Prediction
A Data Science project to predict CLV using RFM, Regression, Clustering, SQL &amp; GenAI




A Data Science project developed during internship at **Celebal Technologies** under the **Data Science Course**. This project predicts the future value a customer will bring to a business throughout their relationship, using transaction data, RFM analysis, clustering, regression, SQL logic, and Generative AI.

---

## 📁 Dataset

- **Name:** Online Retail II  
- **Source:** UCI Machine Learning Repository  
- **Link:** [Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Duration Covered:** Dec 1, 2009 – Dec 9, 2011
- **Description:** Includes all transactions from a UK-based online retail store selling unique giftware to both individuals and wholesalers.

---

## 🎯 Objective

> Predict the **Customer Lifetime Value (CLV)** by analyzing customer behavior such as:
- Frequency of purchases
- Recency and monetary value
- Product returns
- Country and Customer ID segmentation

---

## 🧰 Technologies & Concepts Used

| Category              | Tools/Concepts                             |
|----------------------|--------------------------------------------|
| Language             | Python (Jupyter Notebook / Google Colab)   |
| Data Handling        | Pandas, NumPy                              |
| Visualization        | Seaborn, Matplotlib                        |
| Data Modeling        | Linear Regression, KMeans Clustering       |
| Feature Engineering  | RFM Analysis, One-Hot Encoding             |
| Output Generation    | CSV export (`Final_CLV_Segments.csv`)      |
| Generative AI        | Recommendations based on RFM Segments     |
| SQL Logic            | Integrated via pandas filtering/queries    |
| Git & GitHub         | Version control and project sharing        |

---

## 🔍 Step-by-Step Workflow

1. **Data Loading** – Load data from `online_retail_II.xlsx`
2. **Data Cleaning** – Handle nulls, remove negative quantities (cancellations), and drop duplicates
3. **RFM Analysis** – Calculate Recency, Frequency, and Monetary value
4. **Customer Segmentation** – Use KMeans to segment customers based on RFM scores
5. **CLV Modeling** – Use Linear Regression to predict expected value based on frequency and monetary metrics
6. **Generative AI** – Interpret results and generate strategies for each segment
7. **Final Output** – Export results to `Final_CLV_Segments.csv`

---

## 📸 Sample Output

```

## CustomerID  |  RFM\_Score  |  Cluster  |  Predicted\_CLV  |  Segment

17850       |     421     |     1     |     £543.20      |  High-Value
13089       |     133     |     0     |     £45.10       |  Low-Value
...

```

---

## 📂 Project Structure

```

Customer Lifetime Value Prediction\_Project/
│
├── online\_retail\_II.xlsx            # Dataset
├── Final\_CLV\_Segments.csv           # Output CSV
├── CLV\_Modeling.ipynb               # Jupyter/Colab notebook
├── requirements.txt                 # Dependencies
└── README.md                        # Project description (this file)

````

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/PranavKrSingh/Customer-Lifetime-Value-Prediction.git
cd Customer-Lifetime-Value-Prediction
````

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

* Open `CLV_Modeling.ipynb` in Jupyter/Colab

---

## ✅ Outcomes

* Built a predictive model for Customer Lifetime Value
* Segmented users into **High**, **Medium**, and **Low** CLV groups
* Suggested personalized marketing strategies
* Incorporated Data Science topics like regression, clustering, feature engineering, and GenAI

---

## 👨‍💼 Developed By

**Pranav Kumar Singh**
B.Tech Computer Science Engineering (2021–2025)
Data Science Intern @ Celebal Technologies

---

## 📬 Contact

📧 Email: [pranavkrsingh.cs@gmail.com](mailto:pranavkrsingh.cs@gmail.com)
🌐 GitHub: [PranavKrSingh](https://github.com/PranavKrSingh)

---

````

---

## 🧠 Pro Tips:
- 📷 You can add a **screenshot** of your final CSV or notebook output if you'd like. Place it in your repo and embed it like:
  ```markdown
  ![Output Preview](screenshot.png)
````



