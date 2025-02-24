import tensorflow as tf
import numpy as np
import os
import pickle

# Cargar los datos de entrenamiento
def cargar_datos(archivo):
    preguntas = []
    respuestas = []
    with open(archivo, 'r', encoding='utf-8') as file:
        for line in file:
            pregunta, respuesta = line.strip().split(';')
            preguntas.append(pregunta)
            respuestas.append(int(respuesta))  # Convertir respuestas a enteros
    return preguntas, np.array(respuestas)

# Preprocesar los datos
def preprocesar_datos(preguntas, respuestas):
    tokenizer = tf.keras.layers.TextVectorization()
    tokenizer.adapt(preguntas)
    x_train = tokenizer(preguntas)
    y_train = respuestas
    return x_train, y_train, tokenizer

# Construir el modelo LSTM
def construir_modelo(tokenizer, num_clases=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()) + 1, output_dim=16, mask_zero=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entrenar el modelo
def entrenar_modelo(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=20, batch_size=5, verbose=True)

# Guardar el modelo
def guardar_modelo(model, path='models/modelo.keras'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f'✅ Modelo entrenado y guardado en {path} 🚀')

# Probar el chatbot
def probar_chatbot(model, tokenizer):
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            break
        pregunta_vectorizada = tokenizer([pregunta])
        respuesta = model.predict(pregunta_vectorizada)
        indice = np.argmax(respuesta)
        categorias = ["NEUTRO", "POSITIVO", "NEGATIVO"]
        print("Respuesta: ", categorias[indice])

if __name__ == "__main__":
    archivo_datos = "data/prueba.txt"  # Asegúrate de tener un archivo con preguntas y respuestas
    preguntas, respuestas = cargar_datos(archivo_datos)
    x_train, y_train, tokenizer = preprocesar_datos(preguntas, respuestas)
    modelo = construir_modelo(tokenizer)
    entrenar_modelo(modelo, x_train, y_train)
    guardar_modelo(modelo)
    import pickle

    # Guardar el tokenizer
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    # Descomenta para probar el chatbot
    # probar_chatbot(modelo, tokenizer)
