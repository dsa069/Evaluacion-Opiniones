from googletrans import Translator

# Definir archivos de entrada y salida
archivo_entrada = "dataset_converted.txt"  # Nombre del archivo en inglés
archivo_salida = "dataset_traducido.txt"  # Nombre del archivo traducido

# Crear el traductor
traductor = Translator()

# Leer el archivo de entrada
with open(archivo_entrada, "r", encoding="utf-8") as f:
    lineas = f.readlines()

# Traducir línea por línea
lineas_traducidas = []
for linea in lineas:
    try:
        traduccion = traductor.translate(linea, src="en", dest="es").text
        lineas_traducidas.append(traduccion)
    except Exception as e:
        print(f"Error al traducir: {linea.strip()} - {e}")
        lineas_traducidas.append(linea)  # En caso de error, mantener la línea original

# Guardar el archivo traducido
with open(archivo_salida, "w", encoding="utf-8") as f:
    f.writelines("\n".join(lineas_traducidas))

print(f"Traducción completada. Archivo guardado como '{archivo_salida}'")
