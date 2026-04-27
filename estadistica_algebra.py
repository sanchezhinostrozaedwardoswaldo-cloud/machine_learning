"""
PASOS 2, 3 y 4 — Iris Dataset
  PASO 2: Exploración y limpieza con pandas
  PASO 3: Estadística descriptiva + gráficos
  PASO 4: Álgebra lineal (vectores, matrices, producto punto, sistema de ecuaciones)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────
#  PASO 2: EXPLORACIÓN Y LIMPIEZA
# ──────────────────────────────────────────────────────────────

print("=" * 60)
print("PASO 2: EXPLORACIÓN DEL DATASET")
print("=" * 60)

df = pd.read_csv("Iris.csv")

# Se elimina la columna Id porque no aporta información al modelo
df.drop(columns=["Id"], inplace=True)

print("\n── Primeras filas ──")
print(df.head())

print("\n── Columnas y tipos de datos ──")
print(df.dtypes)

print("\n── Valores nulos por columna ──")
print(df.isnull().sum())

print("\n── Forma del dataset (filas, columnas) ──")
print(df.shape)

print("\n── Distribución de clases ──")
print(df["Species"].value_counts())

# ──────────────────────────────────────────────────────────────
#  PASO 3: ESTADÍSTICA DESCRIPTIVA + GRÁFICOS
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PASO 3: ESTADÍSTICA DESCRIPTIVA")
print("=" * 60)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

for col in features:
    datos = df[col]
    print(f"\n── {col} ──")
    print(f"  Media:              {datos.mean():.4f}")
    print(f"  Mediana:            {datos.median():.4f}")
    print(f"  Varianza:           {datos.var():.4f}")
    print(f"  Desv. estándar:     {datos.std():.4f}")
    print(f"  Mín / Máx:          {datos.min()} / {datos.max()}")

# ── Gráficos ──────────────────────────────────────────────────

colores = {"Iris-setosa": "#4C72B0", "Iris-versicolor": "#DD8452", "Iris-virginica": "#55A868"}

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Iris Dataset — Análisis estadístico", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 1. Histogramas por feature
ax1 = fig.add_subplot(gs[0, 0])
for feat in features:
    ax1.hist(df[feat], bins=15, alpha=0.6, label=feat)
ax1.set_title("Distribución de todas las features")
ax1.set_xlabel("Valor (cm)")
ax1.set_ylabel("Frecuencia")
ax1.legend(fontsize=7)

# 2. Boxplot por especie — PetalLengthCm
ax2 = fig.add_subplot(gs[0, 1])
species_list = df["Species"].unique()
data_box = [df[df["Species"] == sp]["PetalLengthCm"].values for sp in species_list]
bp = ax2.boxplot(data_box, patch_artist=True, labels=[s.split("-")[1] for s in species_list])
for patch, sp in zip(bp["boxes"], species_list):
    patch.set_facecolor(colores[sp])
ax2.set_title("Boxplot — Longitud del pétalo por especie")
ax2.set_ylabel("PetalLengthCm")

# 3. Scatter SepalLength vs PetalLength
ax3 = fig.add_subplot(gs[1, 0])
for sp, color in colores.items():
    subset = df[df["Species"] == sp]
    ax3.scatter(subset["SepalLengthCm"], subset["PetalLengthCm"],
                label=sp.split("-")[1], color=color, alpha=0.7, edgecolors="white", s=50)
ax3.set_title("Sépal Length vs Petal Length")
ax3.set_xlabel("SepalLengthCm")
ax3.set_ylabel("PetalLengthCm")
ax3.legend()

# 4. Matriz de correlación (heatmap manual con imshow)
ax4 = fig.add_subplot(gs[1, 1])
corr = df[features].corr().values
im = ax4.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax4.set_xticks(range(len(features)))
ax4.set_yticks(range(len(features)))
labels_short = ["SepLen", "SepWid", "PetLen", "PetWid"]
ax4.set_xticklabels(labels_short, rotation=30, fontsize=8)
ax4.set_yticklabels(labels_short, fontsize=8)
for i in range(len(features)):
    for j in range(len(features)):
        ax4.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="black" if abs(corr[i, j]) < 0.7 else "white")
ax4.set_title("Matriz de correlación")
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

plt.savefig("estadistica_graficos.png", dpi=150, bbox_inches="tight")
print("\nGráficos guardados en: estadistica_graficos.png")
plt.show()

# ──────────────────────────────────────────────────────────────
#  PASO 4: ÁLGEBRA LINEAL
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PASO 4: ÁLGEBRA LINEAL")
print("=" * 60)

# ── Vectores ──────────────────────────────────────────────────
print("\n── Vectores (primeras dos muestras del dataset) ──")
v1 = np.array(df[features].iloc[0])   # Iris-setosa muestra 0
v2 = np.array(df[features].iloc[50])  # Iris-versicolor muestra 0
print(f"  v1 (setosa)     = {v1}")
print(f"  v2 (versicolor) = {v2}")
print(f"  v1 + v2         = {v1 + v2}")
print(f"  v1 × 2          = {v1 * 2}")
print(f"  Norma ||v1||    = {np.linalg.norm(v1):.4f}")
# ── Matrices ──────────────────────────────────────────────────
print("\n── Matrices (subconjunto 4×4 del dataset) ──")
M = df[features].values[:4]          # 4 filas × 4 columnas
print(f"  M =\n{M}")
print(f"\n  Transpuesta M.T =\n{M.T}")
# ── Producto punto ─────────────────────────────────────────────
print("\n── Producto punto ──")
dot = np.dot(v1, v2)
print(f"  v1 · v2 = {dot:.4f}")
cos_sim = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"  Similitud coseno = {cos_sim:.4f}  (1 = idénticos, 0 = ortogonales)")
# Producto de matrices cuadradas (M^T × M)
MtM = M.T @ M   # 4×4
print(f"\n  M^T × M (4×4) =\n{MtM}")
# ── Sistema de ecuaciones ──────────────────────────────────────
print("\n── Sistema de ecuaciones lineales (Ax = b) ──")
A = np.array([
    [5.1, 4.9, 4.7],
    [3.5, 3.0, 3.2],
    [1.4, 1.4, 1.3]
], dtype=float)
b = np.array([14.7, 9.7, 4.1])

print("  A =")
print(f"  {A}")
print(f"\n  b = {b}")

det = np.linalg.det(A)
print(f"\n  det(A) = {det:.6f}")

if abs(det) > 1e-10:
    x = np.linalg.solve(A, b)
    print(f"  Solución x = {x}")
    print(f"  Verificación Ax = {A @ x}  ≈  b = {b}")
else:
    print("  El sistema no tiene solución única (determinante ≈ 0)")

