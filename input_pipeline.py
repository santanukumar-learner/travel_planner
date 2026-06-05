import pickle
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from data_converter import CrowdPredictionInputProcessor


def main():
    load_dotenv()

    model_path = os.path.join(os.path.dirname(__file__), "rkl_places_xgb.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    processor = CrowdPredictionInputProcessor()

    place = "Hanuman Vatika"
    target_date = datetime(2025, 5, 30, 18, 0)  # 6 PM
    weather_key = os.getenv("WHEATHER_KEY")

    feature_vector = processor.prepare_model_input(
        place_name=place,
        target_datetime=target_date,
        target_hour=target_date.hour,
        weather_api_key=weather_key,
    )

    X = np.array(feature_vector, dtype=float).reshape(1, -1)
    prediction = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
        print("Crowd probability (class 1):", proba)

    context = processor.get_prediction_context(
        place_name=place,
        target_datetime=target_date,
        target_hour=target_date.hour,
        weather_api_key=weather_key,
    )

    print("\n=== Prediction Result ===")
    print("Context:", context)
    print("Feature Vector:", feature_vector)
    print("Model Prediction:", prediction[0])


if __name__ == "__main__":
    main()

