
import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join('data','rfm_features.csv')
MODEL_PATH = os.path.join('models','clv_model.pkl')

# Load data
df = pd.read_csv(DATA_PATH)
X = df[['Recency','Frequency','Monetary','AvgOrderValue']]
y = df['Monetary']  # Simple proxy: using Monetary spend as CLV target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, objective='reg:squarederror')
model.fit(X_train, y_train)

preds = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))
print('R²:', r2_score(y_test, preds))

joblib.dump(model, MODEL_PATH)
print('✅ Model saved to', MODEL_PATH)
