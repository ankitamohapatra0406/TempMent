import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/processed/behavior_dataset.csv")

features = [
    "motion_score",
    "temperature",
    "humidity",
    "csi_variance"
]

X = df[features]
y = df["behavior_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))

joblib.dump(model, "models/behavior_rf.pkl")

print("Behavior model trained and saved.")