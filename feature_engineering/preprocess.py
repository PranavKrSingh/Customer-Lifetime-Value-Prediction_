"""
Preprocess Online Retail II data for CLV modelling
-------------------------------------------------
Creates:
  • data/rfm_features.csv
  • data/clean_transactions.csv
Usage:
  python feature_engineering/preprocess.py \
      --in data/online_retail_II.xlsx \
      --sheet "Year 2010-2011" \
      --out data
"""

import pandas as pd
import argparse, os

# ──────────────────────────────────────────────────────────────
# CLI ARGUMENTS
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Preprocess Online Retail II data for CLV model"
)
parser.add_argument("--in",   dest="input_path",  default="data/online_retail_II.xlsx",
                    help="Path to the Excel file")
parser.add_argument("--sheet", dest="sheet_name", default="Year 2010-2011",
                    help="Sheet name inside the workbook")
parser.add_argument("--out",  dest="output_dir",  default="data",
                    help="Output directory for processed CSVs")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
print(f"📥 Loading sheet '{args.sheet_name}' from {args.input_path}")
df = pd.read_excel(args.input_path, sheet_name=args.sheet_name)
print("Rows loaded:", len(df))

# ──────────────────────────────────────────────────────────────
# BASIC CLEANING
# ──────────────────────────────────────────────────────────────
# 1. Drop rows without Customer ID
df = df.dropna(subset=["Customer ID"]).copy()
print("After dropping NA Customer ID:", len(df))

# 2. Keep only positive quantity & price
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
print("After filtering positive Quantity & Price:", len(df))

# 3. Total price per line
df["TotalPrice"] = df["Quantity"] * df["Price"]

# ──────────────────────────────────────────────────────────────
# RFM FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("Customer ID")
      .agg(
          Recency   = ("InvoiceDate", lambda x: (reference_date - x.max()).days),
          Frequency = ("Invoice", "nunique"),   # 'Invoice' ≈ transaction ID
          Monetary  = ("TotalPrice", "sum")
      )
)

rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

# ──────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

rfm_out  = os.path.join(args.output_dir, "rfm_features.csv")
trans_out = os.path.join(args.output_dir, "clean_transactions.csv")

rfm.to_csv(rfm_out)
df.to_csv(trans_out, index=False)

print(f"✅ Saved RFM features → {rfm_out}")
print(f"✅ Saved cleaned transactions → {trans_out}")
