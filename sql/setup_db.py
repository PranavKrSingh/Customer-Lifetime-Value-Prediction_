
import pandas as pd, sqlite3, os

CLEAN_PATH = os.path.join('data','clean_transactions.csv')
DB_PATH = os.path.join('data','retail.db')

conn = sqlite3.connect(DB_PATH)
pd.read_csv(CLEAN_PATH).to_sql('transactions', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
print('✅ SQLite DB created at', DB_PATH)
