from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_performance_model.pkl")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form["attendance"]),
                float(request.form["study_hours"]),
                float(request.form["previous_marks"]),
                float(request.form["assignment_score"]),
                float(request.form["internal_marks"]),
                float(request.form["sleep_hours"]),
            ]

            model = load_model()
            data = np.array([values])
            prediction = model.predict(data)[0]

            probabilities = model.predict_proba(data)[0]
            class_index = list(model.classes_).index(prediction)
            probability = round(float(probabilities[class_index]) * 100, 2)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
