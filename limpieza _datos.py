import pandas as pd

# 1. Cargar el CSV 
df = pd.read_csv("real_state_crudo.csv")

# 2. Renombrar columnas para no morir en el intento
nuevos_nombres = {
    'X1 transaction date': 'fecha',
    'X2 house age': 'edad',
    'X3 distance to the nearest MRT station': 'distancia_mrt',
    'X4 number of convenience stores': 'tiendas',
    'X5 latitude': 'lat',
    'X6 longitude': 'long',
    'Y house price of unit area': 'precio'
}
df.rename(columns=nuevos_nombres, inplace=True)

# 3. Quitar la columna 'No' que no sirve para la regresión
if 'No' in df.columns:
    df.drop(columns=['No'], inplace=True)

# 4. TRUCO DE LIMPIEZA: Arreglar los números que perdieron el punto
# Si la distancia es mayor a 10,000, la dividimos
df.loc[df['distancia_mrt'] > 10000, 'distancia_mrt'] = df['distancia_mrt'] / 1000

# 5. Manejo de Nulos
nulos = df.isnull().sum()
print("Valores nulos encontrados:\n", nulos)

# Si hubiera nulos, los borramos 
df.dropna(inplace=True)

# 6. Guardar el dataset FINAL
df.to_csv("datos_limpios2.csv", index=False)
print("\n¡Dataset limpio guardado como 'datos_limpios.csv'!")