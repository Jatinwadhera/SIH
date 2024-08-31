import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model (make sure the path to 'model.pkl' is correct)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('INDEX.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("INDEX.html", prediction_text="Recommended crop to cultivate: {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
