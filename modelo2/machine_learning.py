# ======================================================
# PROPÓSITO DEL MODELO
# ======================================================

# Este script implementa un modelo de Machine Learning
# cuyo objetivo es predecir si el mercado bursátil
# subirá o bajará en función de:

# 1. Noticias financieras (texto)
# 2. Sentimiento de las noticias
# 3. Variables financieras del mercado

# Variable objetivo:

# Label
# 0 → El mercado baja
# 1 → El mercado sube

# El modelo intenta encontrar patrones entre:
# noticias + comportamiento financiero del mercado
# para estimar la tendencia futura.


# ======================================================
# LIBRERÍAS
# ======================================================

import pandas as pd
import numpy as np

# División de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Convertir texto en vectores numéricos
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelo de Machine Learning
from sklearn.ensemble import RandomForestClassifier

# Métricas de evaluación
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
# Análisis de sentimiento
from textblob import TextBlob

# Permite unir matrices de texto con variables numéricas
from scipy.sparse import hstack

# Normalización de variables numéricas
from sklearn.preprocessing import StandardScaler

# Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# 1. CARGAR DATASET
# ======================================================

# Se carga el dataset que previamente fue limpiado
# durante la etapa de preprocesamiento de datos.

df = pd.read_csv("dataset_final_limpio.csv")

# Columna con todas las noticias unidas y procesadas
texto = df["texto_limpio"]

# Variable objetivo del modelo
y = df["Label"]


# ======================================================
# 2. CONVERSIÓN DE TEXTO A NÚMEROS (TF-IDF)
# ======================================================

# Los algoritmos de Machine Learning no pueden
# trabajar directamente con texto.

# Por esta razón utilizamos TF-IDF, que convierte
# cada palabra en un valor numérico dependiendo
# de su importancia dentro del documento.

vectorizer = TfidfVectorizer(
    max_features=1000,     # máximo número de palabras a considerar
    ngram_range=(1, 2)     # usa palabras individuales y pares de palabras
)

# Se transforma el texto en una matriz numérica
X_texto = vectorizer.fit_transform(texto)


# ======================================================
# 3. ANÁLISIS DE SENTIMIENTO
# ======================================================

# El análisis de sentimiento permite determinar
# si una noticia tiene una carga positiva o negativa.

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


# Se calcula el sentimiento de cada noticia
df['sentimiento'] = df['texto_limpio'].apply(get_sentiment)

# Convertimos a matriz para poder unirla con TF-IDF
sentimiento_array = df['sentimiento'].values.reshape(-1, 1)


# ======================================================
# 4. VARIABLES FINANCIERAS
# ======================================================

# Además del texto, el modelo utilizará variables
# del comportamiento del mercado.

variables_financieras = df[[
    "Open",
    "High",
    "Low",
    "Volume"
]]

# Estas variables pueden tener escalas muy diferentes,
# por ejemplo:
# volumen puede ser millones mientras precios son pequeños.

# Para evitar que una variable domine al modelo
# se realiza una normalización.

scaler = StandardScaler()

variables_financieras = scaler.fit_transform(variables_financieras)


# ======================================================
# 5. UNIÓN DE TODAS LAS VARIABLES
# ======================================================

# Ahora unimos tres tipos de información:

# 1. Texto vectorizado (TF-IDF)
# 2. Sentimiento de las noticias
# 3. Variables financieras

X = hstack((X_texto, sentimiento_array, variables_financieras))


# ======================================================
# 6. DIVISIÓN TRAIN TEST
# ======================================================

# El dataset se divide en dos partes:

# entrenamiento → para que el modelo aprenda
# prueba → para evaluar qué tan bien predice

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ======================================================
# 7. ENTRENAMIENTO DEL MODELO
# ======================================================

# Se utiliza Random Forest.

# Random Forest funciona creando muchos árboles
# de decisión y combinando sus resultados.

# Esto permite detectar patrones complejos
# entre texto y datos financieros.

"""modelo = RandomForestClassifier(
    n_estimators=200,   # número de árboles
    max_depth=12,       # profundidad de los árboles
    random_state=42
)"""

modelo = LogisticRegression(C=0.1, solver='liblinear')

modelo.fit(X_train, y_train)


# ======================================================
# 8. PREDICCIONES
# ======================================================

# El modelo intenta predecir el comportamiento
# del mercado utilizando los datos de prueba.

pred = modelo.predict(X_test)


# ======================================================
# 9. EVALUACIÓN DEL MODELO
# ======================================================

# Accuracy → porcentaje de predicciones correctas
accuracy = accuracy_score(y_test, pred)

# MSE → error promedio
mse = mean_squared_error(y_test, pred)

# R² → capacidad explicativa del modelo
r2 = r2_score(y_test, pred)

print("Accuracy:", accuracy)
print("MSE:", mse)


# distribución de clases
print(df["Label"].value_counts(normalize=True))


# ======================================================
# 10. MATRIZ DE CONFUSIÓN
# ======================================================

# La matriz de confusión muestra cuántas veces
# el modelo acertó o se equivocó.

matriz = confusion_matrix(y_test, pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")

plt.savefig("matriz_confusion.png", dpi=300)
plt.close()


# ======================================================
# 11. GRÁFICO DE DISPERSIÓN
# ======================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, pred, alpha=0.5)

plt.title("Comparación: Valores Reales vs Predicción")
plt.xlabel("Valor real")
plt.ylabel("Predicción")

plt.savefig("dispersion_prediccion.png", dpi=300)
plt.close()


# ======================================================
# 12. PRUEBA CON UNA NOTICIA NUEVA
# ======================================================

print("\nProbando modelo con noticia nueva")

texto_nuevo = [
    "global economic growth increases corporate profits and market optimism"
]

# vectorización del texto
vector_palabras = vectorizer.transform(texto_nuevo)

# sentimiento
sentimiento_nuevo = get_sentiment(texto_nuevo[0])
sentimiento_array = np.array([[sentimiento_nuevo]])

# ejemplo de variables financieras simuladas
# (solo para probar el modelo)

datos_financieros = np.array([[34000, 34200, 33800, 200000000]])

datos_financieros = scaler.transform(datos_financieros)

# unión de todas las variables
vector_final = hstack((vector_palabras, sentimiento_array, datos_financieros))

resultado = modelo.predict(vector_final)

print("Predicción:", resultado)
