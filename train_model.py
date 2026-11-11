import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "student_performance.csv"
MODEL_FILE = "student_performance_model.pkl"

df = pd.read_csv(DATA_FILE)

features = [
    "Attendance",
    "StudyHours",
    "PreviousMarks",
    "AssignmentScore",
    "InternalMarks",
    "SleepHours"
]
target = "Performance"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_FILE)
print(f"\nModel saved successfully as: {MODEL_FILE}")
