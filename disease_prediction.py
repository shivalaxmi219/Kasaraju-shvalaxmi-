# ============================================================
# DISEASE PREDICTION SYSTEM
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("          DISEASE PREDICTION SYSTEM")
print("=" * 60)

# STEP 1
print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "Fever": [1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
    "Cough": [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0],
    "Headache": [1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1],
    "Fatigue": [1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    "Disease": [
        "Flu","Flu","Cold","Migraine","Flu",
        "Migraine","Flu","Cold","Flu","Migraine",
        "Flu","Migraine","Flu","Cold","Flu",
        "Migraine","Flu","Cold","Flu","Migraine"
    ]
})

print("Dataset collected successfully.")
print("\nFirst 10 records:")
print(data.head(10))
print("\nDataset Shape:", data.shape)

# STEP 2
print("\nSTEP 2: DATA PREPROCESSING")

print("\nMissing Values:")
print(data.isnull().sum())

print("\nData preprocessing completed.")

# STEP 3
print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

print("\nStatistical Summary:")
print(data.describe(include="all"))

print("\nDisease Distribution:")
print(data["Disease"].value_counts())

print("\nAverage Fever:", data["Fever"].mean())
print("Average Cough:", data["Cough"].mean())
print("Average Headache:", data["Headache"].mean())
print("Average Fatigue:", data["Fatigue"].mean())

# STEP 4
print("\nSTEP 4: FEATURE ENGINEERING")

X = data[["Fever", "Cough", "Headache", "Fatigue"]]

disease_mapping = {
    "Cold": 0,
    "Flu": 1,
    "Migraine": 2
}

y = data["Disease"].map(disease_mapping)

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(data["Disease"].head())

print("\nEncoded Target Variable:")
print(y.head())

# STEP 5
print("\nSTEP 5: DATA SPLITTING")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# STEP 6
print("\nSTEP 6: MODEL SELECTION")
print("Random Forest Classifier selected.")

# STEP 7
print("\nSTEP 7: MODEL TRAINING")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")

# STEP 8
print("\nSTEP 8: PREDICTION")

y_pred = model.predict(X_test)

print("Actual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

# STEP 9
print("\nSTEP 9: MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")

# STEP 10
print("\nSTEP 10: CLASSIFICATION REPORT")

print(classification_report(
    y_test,
    y_pred
))

# STEP 11
print("\nSTEP 11: CONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nConfusion Matrix Format:")
print("Rows = Actual")
print("Columns = Predicted")

# STEP 12
print("\nSTEP 12: NEW PATIENT DISEASE PREDICTION")

new_patient = pd.DataFrame({
    "Fever": [1],
    "Cough": [1],
    "Headache": [0],
    "Fatigue": [1]
})

print("\nPatient Details:")
print("Fever     : Yes")
print("Cough     : Yes")
print("Headache  : No")
print("Fatigue   : Yes")

prediction = model.predict(new_patient)

reverse_mapping = {
    0: "Cold",
    1: "Flu",
    2: "Migraine"
}

final_disease = reverse_mapping[prediction[0]]

print("\nFINAL DISEASE PREDICTION:", final_disease)

# STEP 13
print("\nSTEP 13: RESULT")

print("Disease prediction completed successfully.")
print("The system predicted the disease based on symptoms.")

# STEP 14
print("\nSTEP 14: CONCLUSION")

print("Machine Learning was used for disease prediction.")
print("Random Forest Classifier was used for prediction.")
print("The system can help identify possible diseases early.")

print("\n" + "=" * 60)
print("       PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
