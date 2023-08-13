import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.cloud import bigquery

project_id = 'your cloud project ID'

dataset_id = 'your dataset ID'
table_id = 'your table ID'


client = bigquery.Client(project=project_id)

query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
df = client.query(query).to_dataframe()

georgia_data = df[df['place_name'].str.contains('Georgia|GA', case=False)]

georgia_data.dropna(subset=['index_sa'], inplace=True)


X = georgia_data[['yr', 'period']]
y = georgia_data['index_sa']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)
plt.xlabel('Actual HPI')
plt.ylabel('Predicted HPI')
plt.title('Predicted vs Actual HPI in Georgia')
plt.show()

