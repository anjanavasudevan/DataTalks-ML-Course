# Flask app for the churn model
# Import the dependencies

import pickle
from flask import Flask
from flask import request
from flask import jsonify

# Get the model file (the churn prediction model):
model_file = 'model_C=1.bin'

# Load the model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Create the app
app = Flask(__name__)

# Define the route
@app.route('/predict', methods=['POST'])
# Prediction module for demo on the app
def predict():
    """
    Predict whether a customer opts out of service or not using given user data\n
    Output: probability and hard decision variable
    """

    # Get request in json format
    customer_data = request.get_json()

    # Do the actual prediction
    input = dv.transform([customer_data])
    prediction = model.predict_proba(input)[0, 1]
    churn = prediction >= 0.5

    # Prettify the result
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }

    # return the result
    return jsonify(result)


# Main program
if(__name__ == "__main__"):
    app.run(debug=True, host='0.0.0.0', port=9696)
