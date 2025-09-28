from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the pickled model
model = pickle.load(open("Meetings11.pkl", "rb"))

# Define categories
categories = [
    'Training', 'Planning', 'Marketing & Sales', 'Finance', 'HR', 'IT',
    'Operations', 'Logistics', 'Miscellaneous', 'Other'
]

# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def predict_top_3(description):
    # Vectorize the description
    X_new = model.named_steps['vect'].transform([description])

    # Predict probabilities
    probas = model.named_steps['clf'].predict_proba(X_new)[0]

    # Get top 3 indices
    top_3_indices = probas.argsort()[-3:][::-1]

    # Get categories and probabilities
    predictions = []
    for idx in top_3_indices:
        predictions.append({
            'category': categories[idx],
            'probability': float(probas[idx])
        })

    return predictions

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        if not data or "description" not in data:
            return jsonify({"error": "Missing 'description' in request data"}), 400

        description = data.get("description")
        if not isinstance(description, str) or description.strip() == "":
            return jsonify({"error": "'description' must be a non-empty string"}), 400

        prediction_type = data.get("type", "single")

        if prediction_type == "top3":
            predictions = predict_top_3(description)
            return jsonify({
                "type": "top3",
                "predictions": predictions
            })
        else:
            # Single prediction
            prediction = model.predict([description])
            predicted_category = prediction[0]
            return jsonify({
                "type": "single",
                "prediction": predicted_category
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/categories", methods=["GET"])
def get_categories():
    return jsonify({"categories": categories})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
