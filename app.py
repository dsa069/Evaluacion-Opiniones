from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json
import pickle
import os

# Cargar modelo entrenado
modelo_path = "models/modelo.keras"
modelo = tf.keras.models.load_model(modelo_path)

# Cargar tokenizer entrenado - versión mejorada
def load_tokenizer():
    """Load the tokenizer in the most reliable way available"""
    # Try pickle version first (best option)
    try:
        with open("models/tokenizer.pkl", "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, IOError):
        print("No se encontró tokenizer.pkl, intentando con JSON...")
    
    # Try JSON version
    try:
        with open("models/tokenizer.json", "r") as f:
            tokenizer_data = json.load(f)
        
        # Create new TextVectorization layer
        if "config" in tokenizer_data:
            config = tokenizer_data["config"]
            vocab = tokenizer_data["vocabulary"]
            
            # Create new layer with same config
            tokenizer = tf.keras.layers.TextVectorization.from_config(config)
            
            # Set vocabulary
            tokenizer.set_vocabulary(vocab)
            
            print("✅ Tokenizer cargado desde configuración JSON")
            return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer from JSON: {e}")
    
    # Fallback to default
    print("⚠️ Creando tokenizer por defecto")
    return tf.keras.layers.TextVectorization(output_mode='int')

# Load the tokenizer
tokenizer = load_tokenizer()
    
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('app.html')

@app.route('/evaluar', methods=['POST'])
def evaluar_opinion():
    try:
        datos = request.get_json()
        if 'opinion' not in datos:
            return jsonify({"error": "Falta el campo 'opinion'"}), 400

        opinion = datos['opinion']
        
        # Process with TextVectorization layer
        opinion_vectorizada = tokenizer([opinion])
        
        # Get prediction
        prediccion = modelo.predict(opinion_vectorizada)
        
        # Debug info
        print(f"Opinion: {opinion}")
        print(f"Predicción: {prediccion}")
        
        indice = np.argmax(prediccion)
        
        if indice == 1:
            resultado = "POSITIVO"
        elif indice == 0:
            resultado = "NEUTRO"
        elif indice == 2:
            resultado = "NEGATIVO"
        else:
            resultado = "NO ENTIENDO"
        
        # Include debug information
        probabilities = {
            "NEUTRO": float(prediccion[0][0]),
            "POSITIVO": float(prediccion[0][1]),
            "NEGATIVO": float(prediccion[0][2]) if prediccion.shape[1] > 2 else 0.0
        }

        return jsonify({
            "opinion": opinion, 
            "resultado": resultado,
            "probabilities": probabilities
        })
    except Exception as e:
        print(f"Error processing opinion: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)