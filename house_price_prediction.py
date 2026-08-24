# ============================================================
# FAKE NEWS DETECTION SYSTEM
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("              FAKE NEWS DETECTION SYSTEM")
print("=" * 60)


# ============================================================
# STEP 1: DATA COLLECTION
# ============================================================

print("\nSTEP 1: DATA COLLECTION")

data = pd.DataFrame({
    "News": [
        "Government announces new education policy",
        "Scientists discover new method to treat disease",
        "Local school opens new computer laboratory",
        "University announces examination schedule",
        "Government launches new student scholarship",
        "Weather department issues heavy rain warning",
        "Researchers publish study on climate change",
        "City hospital opens new emergency department",
        "Election commission announces voting schedule",
        "Bank introduces new digital payment service",

        "Aliens secretly control the entire government",
        "Drinking water every hour makes people immortal",
        "Scientists confirm humans can live without sleep",
        "Moon will disappear from the sky tomorrow",
        "A magic fruit can cure every disease",
        "Government gives free cars to every citizen",
        "People can become invisible by eating special leaves",
        "Mobile phones can read everyone's thoughts",
        "One coin can make anyone a millionaire overnight",
        "Doctors say exercise is completely unnecessary"
    ],

    "Label": [
        "Real", "Real", "Real", "Real", "Real",
        "Real", "Real", "Real", "Real", "Real",

        "Fake", "Fake", "Fake", "Fake", "Fake",
        "Fake", "Fake", "Fake", "Fake", "Fake"
    ]
})

print("Dataset collected successfully.")

print("\nFirst 10 records:")
print(data.head(10))

print("\nDataset Shape:", data.shape)


# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

print("\nSTEP 2: DATA PREPROCESSING")

print("\nMissing Values:")
print(data.isnull().sum())

# Convert news text to lowercase
data["News"] = data["News"].str.lower()

print("\nText converted to lowercase.")
print("Data preprocessing completed.")


# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")

print("\nNews Distribution:")
print(data["Label"].value_counts())

print("\nAverage News Length:")

data["NewsLength"] = data["News"].apply(len)

print(round(data["NewsLength"].mean(), 2))

print("\nMaximum News Length:")
print(data["NewsLength"].max())

print("\nMinimum News Length:")
print(data["NewsLength"].min())


# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

print("\nSTEP 4: FEATURE ENGINEERING")

X_text = data["News"]
y = data["Label"]

print("\nInput Feature:")
print(X_text.head())

print("\nTarget Variable:")
print(y.head())

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

print("\nTF-IDF feature extraction completed.")
print("Number of features:", X.shape[1])


# ============================================================
# STEP 5: DATA SPLITTING
# ============================================================

print("\nSTEP 5: DATA SPLITTING")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records :", X_test.shape[0])


# ============================================================
# STEP 6: MODEL SELECTION
# ============================================================

print("\nSTEP 6: MODEL SELECTION")

print("Random Forest Classifier selected.")


# ============================================================
# STEP 7: MODEL TRAINING
# ============================================================

print("\nSTEP 7: MODEL TRAINING")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")


# ============================================================
# STEP 8: PREDICTION
# ============================================================

print("\nSTEP 8: PREDICTION")

y_pred = model.predict(X_test)

print("Actual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)


# ============================================================
# STEP 9: MODEL EVALUATION
# ============================================================

print("\nSTEP 9: MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# ============================================================
# STEP 10: CLASSIFICATION REPORT
# ============================================================

print("\nSTEP 10: CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Fake", "Real"],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)


# ============================================================
# STEP 11: CONFUSION MATRIX
# ============================================================

print("\nSTEP 11: CONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Fake", "Real"]
)

print(cm)

print("\nConfusion Matrix Format:")
print("Rows    = Actual")
print("Columns = Predicted")
print("[[True Fake, False Real],")
print(" [False Fake, True Real]]")


# ============================================================
# STEP 12: NEW NEWS PREDICTION
# ============================================================

print("\nSTEP 12: NEW NEWS PREDICTION")

new_news = [
    "Scientists announce a new research study about health"
]

print("\nNews Details:")
print(new_news[0])

new_news_vector = vectorizer.transform(new_news)

prediction = model.predict(new_news_vector)

print("\nFINAL NEWS PREDICTION:", prediction[0])


# ============================================================
# STEP 13: RESULT
# ============================================================

print("\nSTEP 13: RESULT")

print("Fake news detection completed successfully.")
print("The system classified the news using Machine Learning.")


# ============================================================
# STEP 14: CONCLUSION
# ============================================================

print("\nSTEP 14: CONCLUSION")

print("Machine Learning was used for fake news detection.")
print("TF-IDF was used for text feature extraction.")
print("Random Forest Classifier was used for prediction.")
print("The system can help identify potentially fake news.")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("          PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
