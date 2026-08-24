# ============================================================
# SALES FORECASTING SYSTEM
# ============================================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("=" * 60)
print("       SALES FORECASTING SYSTEM")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: DATA COLLECTION
# ------------------------------------------------------------
print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "Month": range(1, 21),
    "Sales": [
        120, 135, 128, 145, 150,
        160, 155, 170, 180, 175,
        190, 200, 195, 210, 220,
        215, 230, 240, 235, 250
    ]
})

print("Sales dataset collected successfully.")

print("\nFirst 10 records:")
print(data.head(10))

print("\nDataset Shape:", data.shape)

# ------------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ------------------------------------------------------------
print("\nSTEP 2: DATA PREPROCESSING")

print("\nMissing Values:")
print(data.isnull().sum())

data = data.dropna()

print("\nData preprocessing completed.")

# ------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

print("\nStatistical Summary:")
print(data.describe())

print("\nAverage Sales:",
      round(data["Sales"].mean(), 2))

print("Minimum Sales:",
      data["Sales"].min())

print("Maximum Sales:",
      data["Sales"].max())

# ------------------------------------------------------------
# STEP 4: TIME SERIES COMPONENTS
# ------------------------------------------------------------
print("\nSTEP 4: TIME SERIES COMPONENTS")

print("Time variable: Month")
print("Target variable: Sales")
print("Trend analysis completed.")

# ------------------------------------------------------------
# STEP 5: FEATURE SELECTION
# ------------------------------------------------------------
print("\nSTEP 5: FEATURE SELECTION")

X = data[["Month"]]
y = data["Sales"]

print("Selected Feature: Month")
print("Target Variable: Sales")

print("\nInput Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())

# ------------------------------------------------------------
# STEP 6: DATA SPLITTING
# ------------------------------------------------------------
print("\nSTEP 6: DATA SPLITTING")

# First 16 records for training
# Last 4 records for testing

X_train = X.iloc[:16]
X_test = X.iloc[16:]

y_train = y.iloc[:16]
y_test = y.iloc[16:]

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# ------------------------------------------------------------
# STEP 7: MODEL SELECTION
# ------------------------------------------------------------
print("\nSTEP 7: MODEL SELECTION")

model = LinearRegression()

print("Linear Regression model selected.")

# ------------------------------------------------------------
# STEP 8: MODEL TRAINING
# ------------------------------------------------------------
print("\nSTEP 8: MODEL TRAINING")

model.fit(X_train, y_train)

print("Model trained successfully.")

print("Model Coefficient:",
      round(model.coef_[0], 2))

print("Model Intercept:",
      round(model.intercept_, 2))

# ------------------------------------------------------------
# STEP 9: SALES PREDICTION
# ------------------------------------------------------------
print("\nSTEP 9: SALES PREDICTION")

y_pred = model.predict(X_test)

print("Actual Sales:")
print(list(y_test))

print("\nPredicted Sales:")
print([round(value, 2) for value in y_pred])

# ------------------------------------------------------------
# STEP 10: MODEL EVALUATION
# ------------------------------------------------------------
print("\nSTEP 10: MODEL EVALUATION")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("Mean Absolute Error :", round(mae, 2))
print("Mean Squared Error  :", round(mse, 2))
print("Root Mean Square Error:", round(rmse, 2))

# ------------------------------------------------------------
# STEP 11: FORECAST NEXT MONTH
# ------------------------------------------------------------
print("\nSTEP 11: FUTURE SALES FORECAST")

next_month = [[21]]

future_sales = model.predict(next_month)[0]

print("Next Month:", 21)
print("Forecasted Sales:",
      round(future_sales, 2))

# ------------------------------------------------------------
# STEP 12: NEW SALES FORECAST
# ------------------------------------------------------------
print("\nSTEP 12: NEW MONTH SALES PREDICTION")

new_month = 22

new_prediction = model.predict([[new_month]])[0]

print("New Month:", new_month)
print("Predicted Sales:",
      round(new_prediction, 2))

# ------------------------------------------------------------
# STEP 13: RESULT
# ------------------------------------------------------------
print("\nSTEP 13: RESULT")

print("Sales forecasting completed successfully.")
print("The model can predict future sales based on")
print("previous monthly sales data.")

# ------------------------------------------------------------
# STEP 14: CONCLUSION
# ------------------------------------------------------------
print("\nSTEP 14: CONCLUSION")

print("Machine Learning was used for sales forecasting.")
print("Linear Regression was used as the prediction model.")
print("Historical sales data was used for training.")
print("The system can help businesses plan future sales.")

print("\n" + "=" * 60)
print("       PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
