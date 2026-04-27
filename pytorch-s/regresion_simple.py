import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 1.Leemos el archivo de nuestros datos limpios
df = pd.read_csv("datos_limpios.csv")

# 2.Definimos las columnas que queremos para X
columnas_x = ['edad', 'distancia_mrt', 'tiendas', 'lat', 'long']
# Definimos la columna pa Y
columnas_y = 'precio'

# 3.Creamos los sub_datasets X y Y extrayendolo de nuestro archivo principal
x = df[columnas_x]
y = df[columnas_y]

# Verificamos que todo salga bien
print("Primeras filas de X (Entradas):")
print(x.head())
print("\nPrimeras filas de y (salida)")
print(y.head())

# 4.Dividimos las variables de entrada y salida
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5.Entrenamos al modelo
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# 6.Ponemos a prueba el modelo
y_pred = modelo.predict(x_test)

# 7.Evaluamos que tan bueno es
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"---RESULTADO SKLEARN---")
print(f"MSE (Error Medio): {mse:.2f}")
print(f"R2 (Precisiòn): {r2:.2f}")

nueva_casa = pd.DataFrame([{
    "edad": 10,
    "distancia_mrt": 500,
    "tiendas": 8,
    "lat": 24.97,
    "long": 121.54
}])

precio_predicho = modelo.predict(nueva_casa)

print(f"\nPrecio estimado de la casa: {precio_predicho[0]:.2f}")

"""# 10.Realizamos una grafica real vs predicho
plt.figure(figsize=(8,6))
# Puntos de predicción
plt.scatter(y_test, y_pred, alpha=0.7)
# Línea de predicción perfecta
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)
# Etiquetas
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
# Título
plt.title("Comparación entre precio real y precio predicho Sklearn")
# Cuadrícula
plt.grid(True)
# Guardar gráfica
plt.savefig("grafica_real_vs_predicho.png")
print("Gráfica guardada como 'grafica_real_vs_predicho.png'")
# Mostrar gráfica
plt.show()"""