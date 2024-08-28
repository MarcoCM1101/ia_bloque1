# Python Libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib

# Flask App
dt = joblib.load('dt1.joblib')

# Create Flask App
server = Flask(__name__)

# Define the route to send the data


@server.route('/predict', methods=['POST'])
def predictjson():
    # Get the data from the request
    data = request.get_json()
    print(data)
    input_data = np.array([
        data['pH'],
        data['sulphates'],
        data['alcohol'],
    ])
    # Predecir utilixando la entrada y el modelo
    result = dt.predict([input_data.reshape(1, -1)])
    # Return the result
    return jsonify({'Prediction': result.str(result[0])})


if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)
