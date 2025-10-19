from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once when API starts
model_path = "/Users/shubham/Desktop/Books/100Days/MLOps/first-project-titantic/models/random_forest_model.pkl"
model = joblib.load(model_path)

class PassengerData(BaseModel):
    Pclass: int
    Sex: int
    Age: int
    Fare: int
    Embarked: int
    Title: int
    IsAlone: int

@app.post("/predict")
def predict(data: PassengerData):
    # Derived feature
    age_class = data.Age * data.Pclass

    # Model input
    sample = np.array([[data.Pclass, data.Sex, data.Age, data.Fare,
                        data.Embarked, data.Title, data.IsAlone, age_class]])

    prediction = model.predict(sample)[0]
    result = "✅ Survived" if prediction == 1 else "❌ Did not survive"
    return {"prediction": result}
