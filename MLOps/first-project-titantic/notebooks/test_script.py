# test_model.py

import joblib
import numpy as np
import os

# ======================
# 1️⃣ Load the model
# ======================
model_path = '/Users/shubham/Desktop/Books/100Days/MLOps/first-project-titantic/models/random_forest_model.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

model = joblib.load(model_path)
print("✅ Model loaded successfully!")


# ======================
# 2️⃣ Define prediction function
# ======================
def predict_survival(Pclass, Sex, Age, Fare, Embarked, Title, IsAlone):
    """
    Predict survival of a Titanic passenger.

    Args:
        Pclass (int): Ticket class (1, 2, 3)
        Sex (int): Male=0, Female=1
        Age (int): Ordinal category (0–4)
        Fare (int): Ordinal category (0–3)
        Embarked (int): S=0, C=1, Q=2
        Title (int): Mr=1, Miss=2, Mrs=3, Master=4, Rare=5
        IsAlone (int): 1 if alone, else 0

    Returns:
        str: Prediction result ("✅ Survived" or "❌ Did not survive")
    """
    # Derived feature
    age_class = Age * Pclass

    # Model input
    sample = np.array([[Pclass, Sex, Age, Fare, Embarked, Title, IsAlone, age_class]])

    # Predict
    prediction = model.predict(sample)[0]
    return "✅ Survived" if prediction == 1 else "❌ Did not survive"


# ======================
# 3️⃣ Test with example passengers
# ======================

# Example 1: Male, 3rd class, young, low fare, from Southampton, alone
result_1 = predict_survival(Pclass=3, Sex=0, Age=1, Fare=0, Embarked=0, Title=1, IsAlone=1)
print(f"Passenger 1 → {result_1}")

# Example 2: Female, 1st class, young, high fare, from Cherbourg, not alone
result_2 = predict_survival(Pclass=1, Sex=1, Age=1, Fare=3, Embarked=1, Title=3, IsAlone=0)
print(f"Passenger 2 → {result_2}")

# Example 3: Elderly male, 2nd class, medium fare, from Queenstown, alone
result_3 = predict_survival(Pclass=2, Sex=0, Age=3, Fare=2, Embarked=2, Title=1, IsAlone=1)
print(f"Passenger 3 → {result_3}")

print("\n✅ All predictions completed successfully!")
