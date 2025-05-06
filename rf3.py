

import itertools
from random_forest import RandomForestRegressor
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


file_path = 'data/selection_test.csv'
data = pd.read_csv(file_path)



feature_columns = [col for col in data.columns if col != 'pathogen load']


X = data[feature_columns]


y = data['pathogen load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(max_depth=30, n_estimators=50)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(y_test)
print(y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")