import pickle
import pandas as pd
import os

# clear screen
os.system('cls' if os.name == 'nt' else 'clear')

def center(text, width=60):
    return text.center(width)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

print("=" * 60)
print(center("AI WEIGHT ESTIMATOR"))
print("=" * 60)
print()
print(center("Available objects: apple, banana, orange, bottle, book"))
print()

obj = input("Enter object name: ")
size = float(input("Enter size in cm: "))

try:
    obj_encoded = encoder.transform([obj])[0]

    input_data = pd.DataFrame([[obj_encoded, size]],
                              columns=["object", "size_cm"])

    prediction = model.predict(input_data)

    print()
    print("=" * 60)
    print(center(f"PREDICTED WEIGHT: {round(prediction[0],2)} grams"))
    print("=" * 60)

except:
    print("Invalid input!")