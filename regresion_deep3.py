import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Leer dataset
df = pd.read_csv("datos_limpios.csv")

columnas_x = ["edad","distancia_mrt","tiendas","lat","long"]
columnas_y = "precio"

x = df[columnas_x]
y = df[columnas_y]

# 2. Escalar datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

# 3. Train test split
x_train,x_test,y_train,y_test = train_test_split(
    x_scaled,y_scaled,test_size=0.2,random_state=42
)

# 4. Convertir a tensores
x_train = torch.tensor(x_train,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)

# 5. Crear batches
dataset = torch.utils.data.TensorDataset(x_train,y_train)
loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)

# 6. Red neuronal
class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()

        self.modelo = nn.Sequential(
            nn.Linear(5, 10),  # Capa oculta con 10 neuronas
            nn.ReLU(),  # Función de activación ReLU
            nn.Linear(10, 1), 
        )

    def forward(self,x):
        return self.modelo(x)

modelo = RedNeuronal()

# 7. Configuración entrenamiento
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(),lr=0.01)

# 8. Entrenamiento
for epoch in range(300):

    loss_total = 0

    for xb, yb in loader:

        pred = modelo(xb)
        loss = criterio(pred, yb)

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        loss_total += loss.item()

    loss_promedio = loss_total / len(loader)

    if (epoch+1) % 50 == 0:
        print(f"Epoca {epoch+1} - Loss promedio: {loss_promedio:.4f}")

# 9. Predicción
modelo.eval()

with torch.no_grad():
    predicciones = modelo(x_test)

# Desnormalizar
pred_real = scaler_y.inverse_transform(predicciones.numpy())
real = scaler_y.inverse_transform(y_test.numpy())

# 10. Métricas
mse = mean_squared_error(real,pred_real)
r2 = r2_score(real,pred_real)

print("MSE:",mse)
print("R2:",r2)

"""# 11. Gráfica
plt.figure(figsize=(8,6))

plt.scatter(real,pred_real,alpha=0.7)

plt.plot(
    [real.min(),real.max()],
    [real.min(),real.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Comparación real vs predicho (PyTorch)")
plt.grid(True)

plt.show()"""
