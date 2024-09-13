from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# Cargar el modelo
dt = joblib.load('gb1.joblib')

# Crear Flask App
server = Flask(__name__)

CORS(server)
# CORS(server, resources={
#      r"/*": {"origins": "https://7453-148-241-64-15.ngrok-free.app"}})

# Ruta del archivo CSV donde se almacenarán las predicciones
csv_file_path = 'predicciones.csv'

# Función para guardar predicciones en el CSV


def save_prediction(data, prediction):
    # Verifica si el archivo CSV ya existe
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        # Si el archivo no existe, crea un DataFrame vacío
        df = pd.DataFrame(columns=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                                   'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                                   'Spent', 'Zona', 'Side', 'Prediction'])

    # Agrega la predicción a los datos de entrada
    data['Prediction'] = prediction

    # Crear un DataFrame temporal con la nueva fila
    new_row = pd.DataFrame(data, index=[0])

    # Usar pd.concat() para agregar la nueva fila al DataFrame existente
    df = pd.concat([df, new_row], ignore_index=True)

    # Guarda el DataFrame actualizado en el archivo CSV
    df.to_csv(csv_file_path, index=False)

# Ruta para predecir usando JSON


@server.route('/predict2', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)

        # Crear DataFrame a partir de los datos
        df = pd.DataFrame(data, index=[0])
        print("here", df)

        # Transformar datos (si es necesario, descomentar transform_data)
        # df2 = transform_data(df)
        # print("here2", df2)

        # Realizar la predicción
        result = dt.predict(df)

        # Convertir el resultado en un mensaje de texto
        if result[0] == 1:
            prediction_message = "Se ha transportado"
        elif result[0] == 0:
            prediction_message = "No se ha transportado"
        else:
            prediction_message = "Resultado no válido"

        print(result[0])
        print(prediction_message)

        # Guardar la predicción en el archivo CSV
        save_prediction(data, result[0])

        # Retornar la respuesta JSON
        return jsonify({'Prediction': prediction_message})

    except ValueError as ve:
        # Manejo de errores específicos, como JSON vacío
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        # Manejo general de excepciones
        return jsonify({'error': 'Error en el procesamiento de los datos', 'details': str(e)}), 500


if __name__ == '__main__':
    server.run(debug=True, port=8080)
