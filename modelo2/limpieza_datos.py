# ======================================================
# PROPÓSITO DEL SCRIPT
# ======================================================

# Este script tiene como objetivo preparar los datos
# antes de aplicar técnicas de Machine Learning
# y Deep Learning.

# El dataset contiene titulares de noticias financieras
# relacionadas con el índice Dow Jones (DJIA).

# Cada día tiene 25 titulares de noticias que podrían
# influir en el comportamiento del mercado.

# El objetivo de este proceso es:

# 1. Leer datasets provenientes de fuentes abiertas (Kaggle)
# 2. Limpiar los datos eliminando valores nulos
# 3. Unir múltiples titulares de noticias en un solo texto
# 4. Aplicar técnicas de procesamiento de lenguaje natural (NLP)
# 5. Realizar tokenización del texto
# 6. Eliminar palabras irrelevantes (stopwords)
# 7. Generar un dataset final listo para entrenamiento

# El resultado será un dataset limpio que permitirá
# entrenar modelos capaces de predecir si el mercado
# bursátil subirá o bajará en función de las noticias.


# ======================================================
# LIBRERÍAS
# ======================================================

# Pandas → manipulación de datos
import pandas as pd

# Numpy → operaciones numéricas
import numpy as np

# Procesamiento de lenguaje natural
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Gráficos para exploración inicial
import matplotlib.pyplot as plt
import seaborn as sns


# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# ======================================================
# 1. CARGA DE DATASETS
# ======================================================
# OBJETIVO:
# Leer datasets CSV provenientes de Kaggle.
# Estos datasets contienen noticias financieras y
# datos históricos del índice Dow Jones.

print("Cargando datasets...")

news = pd.read_csv("Combined_News_DJIA.csv")
reddit = pd.read_csv("RedditNews.csv")
stock = pd.read_csv("upload_DJIA_table.csv")

print("\nPrimeras filas del dataset de noticias")
print(news.head())


# ======================================================
# 2. REVISIÓN DE DATOS
# ======================================================
# OBJETIVO:
# Verificar si existen valores nulos o inconsistencias
# en los datos antes de realizar análisis.

print("\nVerificando valores nulos")
print(news.isnull().sum())
# Reemplazamos nulos por texto vacío
news = news.fillna("")


# ======================================================
# 3. UNIÓN DE NOTICIAS
# ======================================================
# OBJETIVO:
# Cada fila tiene 25 titulares de noticias.
# Vamos a unirlos en un solo texto para poder
# aplicar técnicas de procesamiento de lenguaje natural.

print("\nUniendo noticias en una sola columna por dia")
noticias = news.iloc[:,2:27]
news["texto"] = noticias.apply(lambda x: " ".join(x), axis=1)

print(news["texto"].head())


# ======================================================
# 4. PROCESAMIENTO DE TEXTO (NLP)
# ======================================================
# OBJETIVO:
# Transformar texto en una forma útil para machine learning
#
# Pasos:
# - convertir a minúsculas
# - eliminar símbolos
# - tokenización
# - eliminar stopwords

print("\nProcesando texto con NLP")

stop_words = set(stopwords.words("english"))

textos_limpios = []

for texto in news["texto"]:

    # convertir a minúsculas
    texto = texto.lower()

    # eliminar caracteres especiales
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    # tokenización
    tokens = word_tokenize(texto)

    tokens_limpios = []

    for palabra in tokens:

        if palabra not in stop_words:
            tokens_limpios.append(palabra)

    texto_final = " ".join(tokens_limpios)

    textos_limpios.append(texto_final)

news["texto_limpio"] = textos_limpios

print("\nTexto procesado")
print(news["texto_limpio"].head())


# ======================================================
# 5. UNIR CON DATOS DEL MERCADO
# ======================================================
# OBJETIVO:
# Relacionar noticias con el comportamiento del mercado.

print("\nUniendo noticias con datos del mercado")

data_final = pd.merge(news, stock, on="Date")

print(data_final.head())


# ======================================================
# 6. VISUALIZACIÓN INICIAL
# ======================================================
# OBJETIVO:
# Explorar distribución de datos antes del modelado.

plt.figure()

sns.countplot(x=data_final["Label"])

plt.title("Distribución de movimiento del mercado")
plt.xlabel("0 = baja  |  1 = sube")

plt.show()


# ======================================================
# 7. GUARDAR DATASET FINAL
# ======================================================

print("\nGuardando dataset limpio")

data_final.to_csv("dataset_final_limpio.csv", index=False)

print("Dataset listo para modelos")
