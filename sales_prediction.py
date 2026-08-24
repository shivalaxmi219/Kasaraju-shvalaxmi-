# ============================================================
# CUSTOMER SEGMENTATION USING CLUSTERING
# ============================================================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("       CUSTOMER SEGMENTATION SYSTEM")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: DATA COLLECTION
# ------------------------------------------------------------
print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "CustomerID": range(1, 21),
    "Age": [22,25,28,30,35,40,42,45,50,52,
            23,27,32,36,39,44,48,51,55,60],
    "AnnualIncome": [20000,25000,30000,35000,40000,
                     45000,50000,55000,60000,65000,
                     22000,28000,33000,38000,43000,
                     48000,53000,58000,70000,80000],
    "SpendingScore": [85,90,78,75,70,65,60,55,50,45,
                      88,82,76,72,68,62,58,52,40,35]
})

print("Dataset collected successfully.")
print("\nFirst 10 records:")
print(data.head(10))

print("\nDataset Shape:", data.shape)

# ------------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ------------------------------------------------------------
print("\nSTEP 2: DATA PREPROCESSING")

print("\nMissing Values:")
print(data.isnull().sum())

print("\nData preprocessing completed.")

# ------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

print("\nStatistical Summary:")
print(data.describe())

print("\nAverage Age:", round(data["Age"].mean(), 2))
print("Average Annual Income:", round(data["AnnualIncome"].mean(), 2))
print("Average Spending Score:", round(data["SpendingScore"].mean(), 2))

# ------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ------------------------------------------------------------
print("\nSTEP 4: FEATURE ENGINEERING")

features = data[["Age", "AnnualIncome", "SpendingScore"]]

print("\nInput Features:")
print(features.head())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

print("\nFeatures standardized successfully.")

# ------------------------------------------------------------
# STEP 5: DATA SPLITTING
# ------------------------------------------------------------
print("\nSTEP 5: DATA PREPARATION")

print("Total customer records:", len(X_scaled))
print("Features used: Age, AnnualIncome, SpendingScore")
print("Data preparation completed.")

# ------------------------------------------------------------
# STEP 6: MODEL SELECTION
# ------------------------------------------------------------
print("\nSTEP 6: MODEL SELECTION")

model = KMeans(n_clusters=3, random_state=42, n_init=10)

print("K-Means Clustering model selected.")
print("Number of clusters:", 3)

# ------------------------------------------------------------
# STEP 7: MODEL TRAINING
# ------------------------------------------------------------
print("\nSTEP 7: MODEL TRAINING")

model.fit(X_scaled)

print("Model trained successfully.")

# ------------------------------------------------------------
# STEP 8: CUSTOMER SEGMENTATION
# ------------------------------------------------------------
print("\nSTEP 8: CUSTOMER SEGMENTATION")

clusters = model.labels_

data["Cluster"] = clusters

print("\nCustomer Cluster Results:")
print(data[["CustomerID", "Age", "AnnualIncome",
            "SpendingScore", "Cluster"]])

# ------------------------------------------------------------
# STEP 9: CLUSTER EVALUATION
# ------------------------------------------------------------
print("\nSTEP 9: CLUSTER EVALUATION")

print("Number of clusters formed:", len(set(clusters)))
print("Inertia:", round(model.inertia_, 2))

print("Cluster evaluation completed.")

# ------------------------------------------------------------
# STEP 10: CLUSTER SUMMARY
# ------------------------------------------------------------
print("\nSTEP 10: CLUSTER SUMMARY")

summary = data.groupby("Cluster").agg({
    "Age": "mean",
    "AnnualIncome": "mean",
    "SpendingScore": "mean",
    "CustomerID": "count"
})

summary = summary.rename(columns={
    "CustomerID": "NumberOfCustomers"
})

print(summary)

# ------------------------------------------------------------
# STEP 11: CUSTOMER GROUPS
# ------------------------------------------------------------
print("\nSTEP 11: CUSTOMER GROUPS")

for cluster in sorted(data["Cluster"].unique()):
    count = len(data[data["Cluster"] == cluster])
    print("Cluster", cluster, "->", count, "customers")

print("\nCustomer groups created successfully.")

# ------------------------------------------------------------
# STEP 12: NEW CUSTOMER PREDICTION
# ------------------------------------------------------------
print("\nSTEP 12: NEW CUSTOMER SEGMENT PREDICTION")

new_customer = [[35, 42000, 72]]

new_customer_scaled = scaler.transform(new_customer)

new_cluster = model.predict(new_customer_scaled)[0]

print("\nNew Customer Details:")
print("Age              :", new_customer[0][0])
print("Annual Income    :", new_customer[0][1])
print("Spending Score   :", new_customer[0][2])

print("\nPredicted Customer Cluster:", new_cluster)

# ------------------------------------------------------------
# STEP 13: RESULT
# ------------------------------------------------------------
print("\nSTEP 13: RESULT")

print("Customer segmentation completed successfully.")
print("Customers are divided into different groups based")
print("on age, annual income and spending score.")

# ------------------------------------------------------------
# STEP 14: CONCLUSION
# ------------------------------------------------------------
print("\nSTEP 14: CONCLUSION")

print("K-Means Clustering was used for customer segmentation.")
print("Customers were grouped into 3 different segments.")
print("The system helps businesses understand customer behavior.")
print("It can support targeted marketing and business decisions.")

print("\n" + "=" * 60)
print("       PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
