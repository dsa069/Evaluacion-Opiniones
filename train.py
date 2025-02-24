import tensorflow as tf
import numpy as np

# Cargar los datos de entrenamiento
def cargar_datos(archivo):
    preguntas = []
    respuestas = []
    with open(archivo, 'r') as file:
        for line in file:
            pregunta, respuesta = line.strip().split(';')
            preguntas.append(pregunta)
            respuestas.append(respuesta)
    return preguntas, respuestas

# Preprocesar los datos
def preprocesar_datos(preguntas, respuestas):
    tokenizer = tf.keras.layers.TextVectorization()
    tokenizer.adapt(preguntas)
    x_train = tokenizer(preguntas)
    y_train = np.array(respuestas)
    return x_train, y_train, tokenizer

# Construir el modelo LSTM
def construir_modelo(tokenizer, num_clases):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary())+1, output_dim=2, mask_zero=True),
        tf.keras.layers.LSTM(2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Entrenar el modelo
def entrenar_modelo(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=20, batch_size=5, verbose=True)

# Probar el chatbot
def probar_chatbot(model, tokenizer):
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            break
        pregunta = tokenizer([pregunta])
        respuesta = model.predict(pregunta)
        indice = np.argmax(respuesta)
        print(indice)
        if indice == 1:
            R = "POSITIVO"
        elif indice == 0:
            R = "NEUTRO"
        elif indice == 2:
            R = "NEGATIVO"
        else:
            R = "NO ENTIENDO"
        print(R)
