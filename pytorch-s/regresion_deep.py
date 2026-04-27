import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1.Leemos el archivo de nuestros datos limpios
df = pd.read_csv("datos_limpios.csv")

# 2.Definimos las columnas que queremos para X
columnas_x = ["edad", "distancia_mrt", "tiendas", "lat", "long"]
# Definimos la columna pa Y
columnas_y = "precio"

# 3.Creamos los sub_datasets X y Y extrayendolo de nuestro archivo principal
x = df[columnas_x]
y = df[columnas_y]

# 4.Estandarizamos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 5.Dividimos las variables de entrada y salida
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y_scaled, test_size=0.2, random_state=42
)

# 6.Convertir a tensores de Pytorch
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 7.Crear la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(5, 10),  # Capa oculta con 10 neuronas
            nn.ReLU(),  # Función de activación ReLU
            nn.Linear(10, 1),  # Capa de salida
        )

    def forward(self, x):
        return self.modelo(x)


modelo = RedNeuronal()

# 8.Entrenamos el modelo
criterio = nn.MSELoss()  # Error cuadratico medio
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

for epoch in range(500):
    modelo.train()
    salida = modelo(x_train)
    perdida = criterio(salida, y_train)

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoca {epoch+1} - Pérdida: {perdida.item():.4f}")


# 9.Hacer predicciones
modelo.eval()
with torch.no_grad():
    predicciones = modelo(x_test)

# Invertimos la normalizacion para interpretar ls resultados
pred_real = scaler_y.inverse_transform(predicciones.numpy())
real = scaler_y.inverse_transform(y_test.numpy())

# Mostramos los primeros 5 resultados
for i in range(5):
    datos = scaler_x.inverse_transform(x_test[i].reshape(1, -1))[0]
    edad, distancia, tiendas, lat, lon = datos

    print(
        f"Edad: {edad:.2f}, Distancia MRT: {distancia:.2f}, Tiendas: {tiendas:.0f}, "
        f"Lat: {lat:.4f}, Lon: {lon:.4f} -> "
        f"Real: {real[i][0]:.2f} | Predicho: {pred_real[i][0]:.2f}"
    )

mse = mean_squared_error(real, pred_real)
r2 = r2_score(real, pred_real)

print("MSE:", mse)
print("R2:", r2)

# 10.Realizamos una grafica real vs predicho
plt.figure(figsize=(8,6))
# Puntos de predicción
plt.scatter(real, pred_real, alpha=0.7)
# Línea de predicción perfecta
plt.plot(
    [real.min(), real.max()],
    [real.min(), real.max()],
    color="red",
    linewidth=2
)
# Etiquetas
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
# Título
plt.title("Comparación entre precio real y precio predicho Pytorch")
# Cuadrícula
plt.grid(True)
# Guardar gráfica
plt.savefig("grafica_real_vs_predicho2.png")
print("Gráfica guardada como 'grafica_real_vs_predicho2.png'")
# Mostrar gráfica
plt.show()