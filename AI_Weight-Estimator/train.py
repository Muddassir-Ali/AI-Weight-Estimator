import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

print("Loading dataset...")

# Dataset load
data = pd.read_csv("dataset.csv")

print("\nDataset Preview:")
print(data.head())

# Object names کو numbers میں convert کرنا
encoder = LabelEncoder()
data["object"] = encoder.fit_transform(data["object"])

print("\nEncoded Dataset:")
print(data.head())

# Features (Input)
X = data[["object", "size_cm"]]

# Target (Output)
y = data["weight_g"]

# Training اور Testing data split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model create
model = LinearRegression()

# Training
print("\nTraining model...")
model.fit(X_train, y_train)

# Accuracy check
score = model.score(X_test, y_test)

print("\nModel Accuracy:", round(score * 100, 2), "%")

# Model save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print(data["object"].value_counts())
print("\nModel saved successfully!")
print("Files created:")
print("- model.pkl")
print("- encoder.pkl")