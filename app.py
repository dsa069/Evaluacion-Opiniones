from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Cargar modelo entrenado
modelo_path = "models/modelo.keras"
modelo = tf.keras.models.load_model(modelo_path)

# Cargar tokenizer (necesitas guardarlo después del entrenamiento)
import pickle

# Cargar tokenizer entrenado
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


app = Flask(__name__)

@app.route("/")  # Ruta raíz
def home():
    return "¡El servidor Flask está funcionando correctamente! 🚀"

@app.route('/evaluar', methods=['POST'])
def evaluar_opinion():
    datos = request.get_json()
    if 'opinion' not in datos:
        return jsonify({"error": "Falta el campo 'opinion'"}), 400

    opinion = datos['opinion']
    opinion_vectorizada = tokenizer([opinion]).numpy()
    prediccion = modelo.predict(opinion_vectorizada)
    indice = np.argmax(prediccion)

    if indice == 1:
        resultado = "POSITIVO"
    elif indice == 0:
        resultado = "NEUTRO"
    elif indice == 2:
        resultado = "NEGATIVO"
    else:
        resultado = "NO ENTIENDO"

    return jsonify({"opinion": opinion, "resultado": resultado})

if __name__ == '__main__':
    app.run(debug=True)
