
import pandas as pd, os, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join('data','rfm_features.csv')
OUT_PATH  = os.path.join('data','customer_segments.csv')

df = pd.read_csv(DATA_PATH)
X = df[['Recency','Frequency','Monetary']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Segment'] = labels
df.to_csv(OUT_PATH, index=False)
print('✅ Segmentation saved to', OUT_PATH)
