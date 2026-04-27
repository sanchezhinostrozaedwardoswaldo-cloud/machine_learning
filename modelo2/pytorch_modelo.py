# ======================================================
# PROPÓSITO DEL MODELO PYTORCH
# ======================================================

# Este script implementa una red neuronal utilizando
# la librería PyTorch.

# El objetivo del modelo es aprender patrones entre:

# 1. Noticias financieras (texto)
# 2. Variables financieras del mercado
# 3. Movimiento del mercado bursátil

# El modelo intenta predecir si el mercado:

# 0 → bajará
# 1 → subirá

# Para lograrlo se utilizan tres tipos de información:

# 1. Representación TF-IDF del texto
# 2. Variables financieras:
#       Open
#       High
#       Low
#       Volume

# Estas variables permiten que la red neuronal
# no solo analice las noticias, sino también
# el comportamiento real del mercado.

# La red neuronal aprende ajustando sus pesos
# internos mediante el proceso de entrenamiento
# utilizando el algoritmo de optimización Adam
# y la función de pérdida Binary Cross Entropy.


# ======================================================
# LIBRERÍAS
# ======================================================

import pandas as pd
import numpy as np

# Librería de Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim

# Librerías de procesamiento de datos
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
# Permite unir matrices de texto con variables numéricas
from scipy.sparse import hstack
# Análisis de sentimiento
from textblob import TextBlob
# Librerías de visualización
import matplotlib.pyplot as plt

"""
import nltk
nltk.download('vader_lexicon')


# ======================================================
# 1. CARGAR DATASET
# ======================================================

df = pd.read_csv("dataset_final_limpio.csv")

# Texto procesado previamente en el paso de limpieza
texto = df["texto_limpio"]

# Variable objetivo
y = df["Label"]

#Crear la columna de sentimiento
sia = SentimentIntensityAnalyzer()

df["sentimiento"] = df["texto_limpio"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# ======================================================
# 2. TRANSFORMACIÓN TF-IDF
# ======================================================

# TF-IDF convierte palabras en números.
# Esto es necesario porque los modelos
# de Machine Learning no pueden procesar texto directamente.

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Genera una matriz donde cada columna
# representa una palabra importante del dataset

X_texto = vectorizer.fit_transform(texto)


# ======================================================
# 3. VARIABLES FINANCIERAS
# ======================================================

# Se agregan variables del comportamiento
# del mercado bursátil.

variables_financieras = df[[
    "Open",
    "High",
    "Low",
    "Volume",
    "sentimiento"
]]

# Estas variables pueden tener escalas muy distintas.
# Por ejemplo:
# Volume puede ser millones mientras precios son menores.

# Para evitar que una variable domine el modelo
# se realiza una normalización.

scaler = StandardScaler()

variables_financieras = scaler.fit_transform(variables_financieras)


# ======================================================
# 4. UNIR TEXTO Y VARIABLES FINANCIERAS
# ======================================================

# Convertimos la matriz de texto a formato denso
# para poder unirla con las variables financieras.

X_texto = X_texto.toarray()

# Se concatenan horizontalmente las matrices

X = np.hstack((X_texto, variables_financieras))
"""

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
    max_features=5000,     # máximo número de palabras a considerar
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

# convertir matriz sparse a matriz normal
X = X.toarray()

# ======================================================
# 5. TRAIN TEST SPLIT
# ======================================================

# Se divide el dataset en:

# datos de entrenamiento → para que el modelo aprenda
# datos de prueba → para evaluar el modelo

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# ======================================================
# 6. CONVERSIÓN A TENSORES
# ======================================================

# PyTorch trabaja con tensores en lugar de arrays
# de numpy.

X_train = torch.tensor(X_train, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)


# ======================================================
# 7. DEFINICIÓN DE LA RED NEURONAL
# ======================================================

# En PyTorch las redes neuronales se definen
# creando una clase que hereda de nn.Module.

class RedNeuronal(nn.Module):

    def __init__(self):

        super().__init__()

        # Número de entradas del modelo
        # 5000 palabras + 4 variables financieras

        input_size = 5005

        # nn.Sequential permite construir
        # la red capa por capa.

        self.modelo = nn.Sequential(

            # Primera capa
            nn.Linear(input_size,32),

            # Función de activación
            nn.ReLU(),

            # Dropout evita que la red memorice los datos
            nn.Dropout(0.3),

            # Capa de salida
            nn.Linear(32,1),

            # Convierte la salida en probabilidad
            nn.Sigmoid()

        )


    def forward(self,x):

        return self.modelo(x)


# Crear instancia del modelo
modelo = RedNeuronal()


# ======================================================
# 8. FUNCIÓN DE PÉRDIDA
# ======================================================

# Binary Cross Entropy mide el error entre
# la predicción del modelo y el valor real.

criterio = nn.BCELoss()


# ======================================================
# 9. OPTIMIZADOR
# ======================================================

# El optimizador ajusta los pesos de la red
# para reducir el error.

optimizador = optim.Adam(

    modelo.parameters(),
    lr=0.001

)


# ======================================================
# 10. ENTRENAMIENTO
# ======================================================

# Durante el entrenamiento la red neuronal:

# 1. Realiza predicciones
# 2. Calcula el error
# 3. Ajusta los pesos internos
# 4. Repite el proceso varias veces

losses = []

for epoch in range(10):

    # Predicción del modelo
    pred = modelo(X_train)

    # Cálculo del error
    loss = criterio(pred,y_train)

    # Reiniciar gradientes
    optimizador.zero_grad()

    # Backpropagation
    loss.backward()

    # Actualización de pesos
    optimizador.step()

    losses.append(loss.item())

    print("Epoca:",epoch,"Loss:",loss.item())


# ======================================================
# 11. GRÁFICO DEL ENTRENAMIENTO
# ======================================================

plt.plot(losses)

plt.title("Loss entrenamiento PyTorch")

plt.xlabel("Epocas")

plt.ylabel("Error")

plt.savefig("grafico_entrenamiento_pytorch.png", dpi=300)

plt.close()


# ======================================================
# 12. EVALUACIÓN DEL MODELO
# ======================================================

# Se evalúa el modelo utilizando
# los datos que no fueron usados
# durante el entrenamiento.

with torch.no_grad():

    pred_test = modelo(X_test)

# Convertir probabilidades en clases
pred_clases = (pred_test > 0.5).float()

# Calcular accuracy
accuracy = (pred_clases == y_test).sum().item() / len(y_test)

print("Accuracy en test:", accuracy)


# ======================================================
# 13. PRUEBA CON TEXTO NUEVO
# ======================================================

texto_nuevo = [
    "economic crisis causes investors to sell stocks"
]

vector = vectorizer.transform(texto_nuevo)

vector = vector.toarray()

# calcular sentimiento del texto nuevo
# sentimiento
sentimiento_nuevo = get_sentiment(texto_nuevo[0])
sentimiento_array = np.array([[sentimiento_nuevo]])

# incluir sentimiento junto a las variables financieras
datos_financieros = np.array([[34000,34200,33800,200000000]])


datos_financieros = scaler.transform(datos_financieros)

vector_final = np.hstack((vector,sentimiento_array,datos_financieros))

vector_final = torch.tensor(vector_final,dtype=torch.float32)

pred = modelo(vector_final)

print("Predicción:", pred.detach().numpy())
