# ============================================================
# LOAN APPROVAL PREDICTION SYSTEM
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("        LOAN APPROVAL PREDICTION SYSTEM")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: DATA COLLECTION
# ------------------------------------------------------------
print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "ApplicantIncome": [
        5000, 3000, 7000, 2500, 8000,
        4500, 6000, 3500, 9000, 4000,
        7500, 2800, 6500, 5500, 10000,
        3200, 7200, 4800, 8500, 3800
    ],

    "CoapplicantIncome": [
        2000, 1500, 2500, 1000, 3000,
        1800, 2200, 1200, 3500, 1600,
        2800, 900, 2400, 2000, 4000,
        1100, 2600, 1700, 3200, 1300
    ],

    "LoanAmount": [
        200, 180, 250, 150, 300,
        220, 240, 160, 350, 190,
        280, 140, 230, 210, 320,
        170, 270, 200, 310, 180
    ],

    "CreditHistory": [
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 1, 0
    ],

    "Education": [
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 1, 0
    ],

    "SelfEmployed": [
        0, 1, 0, 1, 0,
        0, 1, 0, 0, 1,
        0, 1, 0, 0, 0,
        1, 0, 1, 0, 1
    ],

    "LoanStatus": [
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 1, 0
    ]
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

data = data.dropna()

print("\nData preprocessing completed.")

# ------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

print("\nStatistical Summary:")
print(data.describe())

print("\nAverage Applicant Income:",
      round(data["ApplicantIncome"].mean(), 2))

print("Average Loan Amount:",
      round(data["LoanAmount"].mean(), 2))

print("Average Credit History:",
      round(data["CreditHistory"].mean(), 2))

# ------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ------------------------------------------------------------
print("\nSTEP 4: FEATURE ENGINEERING")

X = data[
    [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "CreditHistory",
        "Education",
        "SelfEmployed"
    ]
]

y = data["LoanStatus"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# ------------------------------------------------------------
# STEP 5: DATA SPLITTING
# ------------------------------------------------------------
print("\nSTEP 5: DATA SPLITTING")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# ------------------------------------------------------------
# STEP 6: MODEL SELECTION
# ------------------------------------------------------------
print("\nSTEP 6: MODEL SELECTION")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Random Forest Classifier selected.")

# ------------------------------------------------------------
# STEP 7: MODEL TRAINING
# ------------------------------------------------------------
print("\nSTEP 7: MODEL TRAINING")

model.fit(X_train, y_train)

print("Model trained successfully.")

# ------------------------------------------------------------
# STEP 8: PREDICTION
# ------------------------------------------------------------
print("\nSTEP 8: PREDICTION")

y_pred = model.predict(X_test)

print("Actual Loan Status:")
print(y_test.values)

print("\nPredicted Loan Status:")
print(y_pred)

# ------------------------------------------------------------
# STEP 9: MODEL EVALUATION
# ------------------------------------------------------------
print("\nSTEP 9: MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# ------------------------------------------------------------
# STEP 10: CLASSIFICATION REPORT
# ------------------------------------------------------------
print("\nSTEP 10: CLASSIFICATION REPORT")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Rejected", "Approved"],
    zero_division=0
))

# ------------------------------------------------------------
# STEP 11: CONFUSION MATRIX
# ------------------------------------------------------------
print("\nSTEP 11: CONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nConfusion Matrix Format:")
print("Rows    = Actual")
print("Columns = Predicted")
print("[[True Rejected, False Approved],")
print(" [False Rejected, True Approved]]")

# ------------------------------------------------------------
# STEP 12: NEW LOAN APPLICATION PREDICTION
# ------------------------------------------------------------
print("\nSTEP 12: NEW LOAN APPLICATION PREDICTION")

new_applicant = pd.DataFrame({
    "ApplicantIncome": [7000],
    "CoapplicantIncome": [2500],
    "LoanAmount": [250],
    "CreditHistory": [1],
    "Education": [1],
    "SelfEmployed": [0]
})

prediction = model.predict(new_applicant)

print("\nApplicant Details:")
print("Applicant Income   : 7000")
print("Coapplicant Income : 2500")
print("Loan Amount        : 250")
print("Credit History     : Good")
print("Education          : Graduate")
print("Self Employed      : No")

if prediction[0] == 1:
    result = "APPROVED"
else:
    result = "REJECTED"

print("\nFINAL LOAN PREDICTION:", result)

# ------------------------------------------------------------
# STEP 13: RESULT
# ------------------------------------------------------------
print("\nSTEP 13: RESULT")

print("Loan approval prediction completed successfully.")

# ------------------------------------------------------------
# STEP 14: CONCLUSION
# ------------------------------------------------------------
print("\nSTEP 14: CONCLUSION")

print("Machine Learning was used to predict loan approval.")
print("Random Forest Classifier was used for prediction.")
print("The system helps in making faster loan approval decisions.")

print("\n" + "=" * 60)
print("         PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
