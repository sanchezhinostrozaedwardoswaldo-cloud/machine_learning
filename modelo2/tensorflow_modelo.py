# ======================================================
# PROPÓSITO DEL MODELO TENSORFLOW
# ======================================================

# Este script implementa una red neuronal utilizando
# TensorFlow y Keras.

# El objetivo del modelo es aprender patrones entre:

# 1. Noticias financieras (texto)
# 2. Sentimiento de las noticias
# 3. Variables financieras del mercado

# Con esta información el modelo intenta predecir
# si el mercado bursátil:

# 0 → bajará
# 1 → subirá

# Este modelo utiliza el mismo enfoque aplicado
# anteriormente con PyTorch para poder comparar
# el rendimiento entre ambos frameworks.


# ======================================================
# LIBRERÍAS
# ======================================================

import pandas as pd
import numpy as np

# Framework de Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input

# Librerías de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Librería para análisis de sentimiento
from textblob import TextBlob

# Librería para unir matrices
from scipy.sparse import hstack

# Librería para graficar
import matplotlib.pyplot as plt


# ======================================================
# 1. CARGAR DATASET
# ======================================================

# Se carga el dataset previamente limpiado
# durante la fase de preprocesamiento.

df = pd.read_csv("dataset_final_limpio.csv")

# Columna que contiene las noticias procesadas
texto = df["texto_limpio"]

# Variable objetivo
y = df["Label"]


# ======================================================
# 2. CONVERSIÓN DE TEXTO A NÚMEROS (TF-IDF)
# ======================================================

# Los algoritmos de Machine Learning y Deep Learning
# no pueden procesar texto directamente.

# Por ello se utiliza TF-IDF que convierte cada palabra
# en un valor numérico dependiendo de su importancia.

vectorizer = TfidfVectorizer(

    max_features=1000,     # máximo número de palabras
    ngram_range=(1,2)      # palabras individuales y pares de palabras

)

# Se genera una matriz donde cada columna representa
# una palabra importante dentro del dataset.

X_texto = vectorizer.fit_transform(texto)


# ======================================================
# 3. ANÁLISIS DE SENTIMIENTO
# ======================================================

# El análisis de sentimiento permite identificar si
# una noticia tiene un tono positivo o negativo.

# Se define una función que calcula el sentimiento.

def get_sentiment(text):

    return TextBlob(text).sentiment.polarity


# Se calcula el sentimiento de cada noticia

df["sentimiento"] = df["texto_limpio"].apply(get_sentiment)

# Convertimos a matriz para poder unirla con otras variables

sentimiento_array = df["sentimiento"].values.reshape(-1,1)


# ======================================================
# 4. VARIABLES FINANCIERAS
# ======================================================

# El modelo también utiliza variables reales
# del comportamiento del mercado.

variables_financieras = df[[
    "Open",
    "High",
    "Low",
    "Volume"
]]

# Estas variables pueden tener escalas muy distintas.
# Por ejemplo:
# Volume puede tener valores de millones mientras
# los precios son mucho menores.

# Para evitar que una variable domine el modelo
# se aplica una normalización.

scaler = StandardScaler()

variables_financieras = scaler.fit_transform(variables_financieras)


# ======================================================
# 5. UNIÓN DE TODAS LAS VARIABLES
# ======================================================

# Ahora combinamos tres fuentes de información:

# 1. TF-IDF del texto
# 2. Sentimiento de la noticia
# 3. Variables financieras

X = hstack((X_texto, sentimiento_array, variables_financieras))

# TensorFlow no trabaja con matrices sparse,
# por lo que se convierte a matriz normal.

X = X.toarray()


# ======================================================
# 6. TRAIN TEST SPLIT
# ======================================================

# Se divide el dataset en:

# entrenamiento → para que el modelo aprenda
# prueba → para evaluar su rendimiento

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# ======================================================
# 7. DEFINICIÓN DE LA RED NEURONAL
# ======================================================

# Sequential permite construir la red capa por capa.

model = Sequential()

# Capa de entrada: 1005 features (1000 TF-IDF + 1 sentimiento + 4 financieras)
model.add(Input(shape=(1005,)))

# Capa oculta con activación ReLU
model.add(Dense(16, activation='relu'))

# Dropout para evitar sobreajuste
model.add(Dropout(0.3))

# Capa de salida: probabilidad entre 0 y 1
model.add(Dense(1, activation='sigmoid'))


# ======================================================
# 8. COMPILACIÓN DEL MODELO
# ======================================================

# Se define:

# optimizador → Adam
# función de pérdida → Binary Crossentropy
# métrica → accuracy

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# ======================================================
# 9. ENTRENAMIENTO
# ======================================================

# Durante el entrenamiento el modelo:

# 1. Realiza predicciones
# 2. Calcula el error
# 3. Ajusta los pesos internos
# 4. Repite el proceso varias veces

# Entrenamiento del modelo durante 5 épocas
historial = model.fit(X_train,y_train,epochs=5,batch_size=32)


# ======================================================
# 10. GRÁFICO DEL ENTRENAMIENTO
# ======================================================

plt.plot(historial.history["loss"])
plt.title("Loss entrenamiento TensorFlow")
plt.xlabel("Epocas")
plt.ylabel("Error")
plt.savefig("grafico_entrenamiento_tensorflow.png", dpi=300)
plt.close()


# ======================================================
# 11. EVALUACIÓN DEL MODELO
# ======================================================

# Se evalúa el rendimiento del modelo
# utilizando datos que no fueron usados
# durante el entrenamiento.

loss, accuracy = model.evaluate(X_test, y_test)

print('Loss:',     loss)
print('Accuracy:', accuracy)


# ======================================================
# 12. PRUEBA CON TEXTO NUEVO
# ======================================================

texto_nuevo = [

    "economic crisis causes investors to sell stocks"

]

# Convertir texto a TF-IDF
vector = vectorizer.transform(texto_nuevo).toarray()

# Calcular sentimiento
sentimiento_nuevo = get_sentiment(texto_nuevo[0])

sentimiento_array = np.array([[sentimiento_nuevo]])

# Variables financieras simuladas

datos_financieros = np.array([[34000,34200,33800,200000000]])

# Normalización

datos_financieros = scaler.transform(datos_financieros)

# Unir todas las variables

vector_final = np.hstack((vector,sentimiento_array,datos_financieros))

# Predicción

pred = model.predict(vector_final)

print("Predicción:",pred)
