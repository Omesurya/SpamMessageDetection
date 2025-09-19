from flask import Flask, request, jsonify, render_template
import pickle

# Load saved model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # frontend page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Transform input
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

        result = "Ham (Not Spam)" if prediction == 1 else "Spam"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
