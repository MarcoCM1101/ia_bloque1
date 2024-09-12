from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Flask App
dt = joblib.load('gb1.joblib')

# Create Flask App
server = Flask(__name__)

# Función para transformar datos


def transform_data(X_test):
    X_test['HomePlanet'] = X_test['HomePlanet'].astype('category').cat.codes
    X_test['Side'] = X_test['Side'].astype('category').cat.codes
    X_test['VIP'] = X_test['VIP'].astype('category').cat.codes
    X_test['Destination'] = X_test['Destination'].astype('category').cat.codes
    X_test['CryoSleep'] = X_test['CryoSleep'].astype('category').cat.codes

    print("aqui", X_test)

    return X_test

# Ruta para predecir usando JSON


@server.route('/predict2', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Crear DataFrame a partir de los datos
        df = pd.DataFrame(data, index=[0])

        # Transformar datos
        df2 = transform_data(df)

        result = dt.predict(df)

        return jsonify({'Prediction': str(result[0])})

    except ValueError as ve:
        # Manejo de errores específicos, como JSON vacío
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        # Manejo general de excepciones
        return jsonify({'error': 'Error en el procesamiento de los datos', 'details': str(e)}), 500


if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=8080)
