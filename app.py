from flask import Flask, render_template, request, jsonify # type: ignore
import tensorflow as tf
import json
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo y el tokenizer
def cargar_modelo():
    modelo_path = 'models/modelo.keras'
    tokenizer_path = 'models/tokenizer.json'
    
    # Cargar el modelo
    if os.path.exists(modelo_path):
        modelo = tf.keras.models.load_model(modelo_path)
        print("✅ Modelo cargado correctamente")
    else:
        raise FileNotFoundError(f"No se encuentra el archivo del modelo en {modelo_path}")
    
    # Cargar configuración del tokenizer
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Recrear el TextVectorization layer
        vocabulary = tokenizer_data['vocabulary']
        config = tokenizer_data['config']
        
        tokenizer = tf.keras.layers.TextVectorization.from_config(config)
        tokenizer.adapt(['placeholder'])  # Necesario para inicializar
        tokenizer.set_vocabulary(vocabulary)
        
        print("✅ Tokenizer cargado correctamente")
    else:
        raise FileNotFoundError(f"No se encuentra el archivo del tokenizer en {tokenizer_path}")
    
    return modelo, tokenizer

# Clasificar el texto
def clasificar_texto(texto, modelo, tokenizer):
    texto_vectorizado = tokenizer([texto])
    prediccion = modelo.predict(texto_vectorizado)
    indice = np.argmax(prediccion)
    categorias = ["NEUTRO", "POSITIVO", "NEGATIVO"]
    return categorias[indice], float(prediccion[0][indice])

# Cargar modelo y tokenizer al iniciar la aplicación
modelo, tokenizer = cargar_modelo()

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/evaluar', methods=['POST'])
def evaluar():
    print("Request received:", request.method)
    print("Content-Type:", request.headers.get('Content-Type'))
    print("Is JSON?", request.is_json)
    
    try:
        # Force JSON parsing regardless of Content-Type
        if request.data:
            try:
                data = json.loads(request.data.decode('utf-8'))
                print("JSON data:", data)
                if 'opinion' in data:
                    texto = data['opinion']
                    print(f"Using opinion from JSON: '{texto}'")
                else:
                    texto = ""
            except json.JSONDecodeError:
                print("Failed to parse JSON")
                texto = ""
        else:
            # Fallback to form data
            texto = request.form.get('texto', '')
            print(f"Using form data: '{texto}'")
        
        if not texto:
            print("No text provided")
            return jsonify({"error": "No se proporcionó texto para evaluar"})
        
        categoria, confianza = clasificar_texto(texto, modelo, tokenizer)
        print(f"Classification result: {categoria} ({confianza})")
        
        return jsonify({
            "opinion": texto,  # Changed to match frontend
            "resultado": categoria,  # Changed to match frontend
            "confianza": round(confianza * 100, 2)
        })
    except Exception as e:
        import traceback
        print(f"Error al evaluar: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)