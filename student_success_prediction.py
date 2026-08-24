# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("      STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: DATA COLLECTION
# ------------------------------------------------------------
print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "StudyHours": [2,5,1,6,3,7,2,8,4,6,
                   3,5,7,1,4,6,8,2,5,7],
    
    "Attendance": [60,85,50,90,70,95,65,98,75,88,
                   58,80,92,55,72,86,96,62,78,94],
    
    "PreviousMarks": [45,75,35,80,55,88,50,92,65,78,
                      48,70,85,40,60,82,95,52,68,90],
    
    "IncomeLevel": [0,1,0,2,1,2,0,2,1,1,
                    0,1,2,0,1,2,2,0,1,2],
    
    "Result": [0,1,0,1,0,1,0,1,1,1,
               0,1,1,0,1,1,1,0,1,1]
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

print("\nAverage Study Hours:", data["StudyHours"].mean())
print("Average Attendance:", data["Attendance"].mean())
print("Average Previous Marks:", data["PreviousMarks"].mean())

# ------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ------------------------------------------------------------
print("\nSTEP 4: FEATURE ENGINEERING")

X = data[["StudyHours", "Attendance", "PreviousMarks", "IncomeLevel"]]
y = data["Result"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# ------------------------------------------------------------
# STEP 5: DATA SPLITTING
# ------------------------------------------------------------
print("\nSTEP 5: DATA SPLITTING")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

print("Actual Results:")
print(y_test.values)

print("\nPredicted Results:")
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
    target_names=["Fail", "Pass"],
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
print("[[True Fail, False Pass],")
print(" [False Fail, True Pass]]")

# ------------------------------------------------------------
# STEP 12: NEW STUDENT PREDICTION
# ------------------------------------------------------------
print("\nSTEP 12: NEW STUDENT PREDICTION")

new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [90],
    "PreviousMarks": [80],
    "IncomeLevel": [1]
})

prediction = model.predict(new_student)

print("\nStudent Details:")
print("Study Hours    : 6")
print("Attendance     : 90%")
print("Previous Marks : 80")
print("Income Level   : Medium")

if prediction[0] == 1:
    result = "PASS"
else:
    result = "FAIL"

print("\nFINAL PREDICTION:", result)

# ------------------------------------------------------------
# STEP 13: RESULT
# ------------------------------------------------------------
print("\nSTEP 13: RESULT")

print("Student performance prediction completed successfully.")

# ------------------------------------------------------------
# STEP 14: CONCLUSION
# ------------------------------------------------------------
print("\nSTEP 14: CONCLUSION")

print("Machine Learning was used to predict student performance.")
print("Random Forest Classifier was used for prediction.")
print("The system can help identify student performance early.")

print("\n" + "=" * 60)
print("       PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
