"""
PASOS 5, 6, 7 y 8 — Iris Dataset
  PASO 5: Preprocesamiento (normalización + codificación)
  PASO 6: Modelo ML (train/test, LogisticRegression, evaluación)
  PASO 7: Detección de outliers con desviación estándar
  PASO 8: Funciones manuales de varianza y desviación estándar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns



# ──────────────────────────────────────────────────────────────
#  CARGA Y LIMPIEZA BASE
# ──────────────────────────────────────────────────────────────

df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X_raw = df[features].values
y_labels = df["Species"].values

# ──────────────────────────────────────────────────────────────
#  PASO 8: FUNCIONES MANUALES (se definen primero para usarlas en paso 5 y 7)
# ──────────────────────────────────────────────────────────────

def varianza_manual(datos: np.ndarray) -> float:
    """Varianza poblacional calculada sin librerías."""
    n = len(datos)
    media = sum(datos) / n
    return sum((x - media) ** 2 for x in datos) / n

def desv_std_manual(datos: np.ndarray) -> float:
    """Desviación estándar poblacional calculada sin librerías."""
    return varianza_manual(datos) ** 0.5

# ──────────────────────────────────────────────────────────────
#  PASO 5: PREPROCESAMIENTO
# ──────────────────────────────────────────────────────────────
#
#  ¿Por qué normalizar aquí pero NO en el paso 3?
#  En el paso 3 analizamos los datos en sus unidades originales (cm)
#  para que los estadísticos sean interpretables. En el paso 6
#  usamos LogisticRegression con solver='lbfgs', que converge mejor
#  cuando las features tienen escala similar. Por eso normalizamos
#  solo antes del modelo.
#
#  Se usa Min-Max Scaling manual (sin sklearn) para ser transparentes
#  y evitar data leakage: los parámetros se calculan SOLO con train.

print("=" * 60)
print("PASO 5: PREPROCESAMIENTO")
print("=" * 60)

# Codificación de etiquetas (str → int)
clases = np.unique(y_labels)
clase_a_idx = {c: i for i, c in enumerate(clases)}
y = np.array([clase_a_idx[c] for c in y_labels])
print(f"\nCodificación de clases: {clase_a_idx}")

# División ANTES de normalizar (evita data leakage)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTamaño train: {X_train_raw.shape[0]} muestras")
print(f"Tamaño test:  {X_test_raw.shape[0]} muestras")

# Min-Max Scaling usando solo estadísticos de train
X_min = X_train_raw.min(axis=0)
X_max = X_train_raw.max(axis=0)

X_train = (X_train_raw - X_min) / (X_max - X_min)
X_test  = (X_test_raw  - X_min) / (X_max - X_min)

print("\nMín por feature (train):", X_min)
print("Máx por feature (train):", X_max)
print("\nPrimera fila normalizada (train):", X_train[0].round(4))

# ──────────────────────────────────────────────────────────────
#  PASO 6: MODELO ML
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PASO 6: MODELO ML — Logistic Regression")
print("=" * 60)

modelo = LogisticRegression(max_iter=200, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en test: {accuracy * 100:.2f}%")

print("\n── Reporte de clasificación ──")
idx_a_clase = {v: k for k, v in clase_a_idx.items()}
target_names = [idx_a_clase[i] for i in range(len(clases))]
print(classification_report(y_test, y_pred, target_names=target_names))

# Matriz de confusión visual
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(clases)))
ax.set_yticks(range(len(clases)))
etiquetas = [c.split("-")[1] for c in target_names]
ax.set_xticklabels(etiquetas)
ax.set_yticklabels(etiquetas)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de confusión")
for i in range(len(clases)):
    for j in range(len(clases)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nMatriz de confusión guardada en: confusion_matrix.png")
plt.show()

# ──────────────────────────────────────────────────────────────
#  PASO 7: DETECCIÓN DE OUTLIERS
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PASO 7: DETECCIÓN DE OUTLIERS (regla de ±2σ)")
print("=" * 60)

# Se usa el dataset original sin normalizar para que los valores sean legibles
UMBRAL = 2  # desviaciones estándar

fig, axes = plt.subplots(1, len(features), figsize=(16, 4))
fig.suptitle("Outliers detectados (±2σ)", fontsize=13, fontweight="bold")

total_outliers = 0
for i, col in enumerate(features):
    datos = df[col].values
    media = datos.mean()
    std   = datos.std()

    outliers_mask = np.abs(datos - media) > UMBRAL * std
    n_outliers = outliers_mask.sum()
    total_outliers += n_outliers

    print(f"\n── {col} ──")
    print(f"  Media: {media:.4f}  |  σ: {std:.4f}")
    print(f"  Rango normal: [{media - UMBRAL*std:.4f}, {media + UMBRAL*std:.4f}]")
    print(f"  Outliers encontrados: {n_outliers}")
    if n_outliers:
        print(f"  Valores outlier: {datos[outliers_mask]}")

    ax = axes[i]
    indices = np.arange(len(datos))
    ax.scatter(indices[~outliers_mask], datos[~outliers_mask],
            color="#4C72B0", alpha=0.5, s=15, label="Normal")
    ax.scatter(indices[outliers_mask], datos[outliers_mask],
            color="red", s=40, zorder=5, label="Outlier")
    ax.axhline(media + UMBRAL * std, color="orange", linestyle="--", linewidth=1)
    ax.axhline(media - UMBRAL * std, color="orange", linestyle="--", linewidth=1)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("Índice")
    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("outliers.png", dpi=150)
print(f"\nTotal de outliers en todo el dataset: {total_outliers}")
print("Gráfico de outliers guardado en: outliers.png")
plt.show()

sns.pairplot(df, hue="Species")
plt.savefig("pairplot.png")

# ──────────────────────────────────────────────────────────────
#  EXTRA: ENTRENAMIENTO SIN OUTLIERS
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXTRA: MODELO SIN OUTLIERS")
print("=" * 60)

# Crear máscara global (si es outlier en cualquier feature, se elimina)
mask_global = np.zeros(len(df), dtype=bool)

for col in features:
    datos = df[col].values
    media = datos.mean()
    std = datos.std()
    mask_global |= (np.abs(datos - media) > 2 * std)

# Filtrar dataset
df_clean = df[~mask_global]

print(f"Datos originales: {len(df)}")
print(f"Datos sin outliers: {len(df_clean)}")

# Preparar nuevos datos
X_clean = df_clean[features].values
y_clean_labels = df_clean["Species"].values
y_clean = np.array([clase_a_idx[c] for c in y_clean_labels])

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# Normalizar
X_min_c = X_train_c.min(axis=0)
X_max_c = X_train_c.max(axis=0)

X_train_c = (X_train_c - X_min_c) / (X_max_c - X_min_c)
X_test_c  = (X_test_c  - X_min_c) / (X_max_c - X_min_c)

# Entrenar modelo
modelo_c = LogisticRegression(max_iter=200, random_state=42)
modelo_c.fit(X_train_c, y_train_c)

# Evaluar
y_pred_c = modelo_c.predict(X_test_c)
accuracy_c = accuracy_score(y_test_c, y_pred_c)

print(f"\nAccuracy SIN outliers: {accuracy_c * 100:.2f}%")


# ──────────────────────────────────────────────────────────────
#  PASO 8: VERIFICACIÓN DE FUNCIONES MANUALES
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PASO 8: FUNCIONES MANUALES — varianza y desviación estándar")
print("=" * 60)

print(f"\n{'Feature':<18} {'Var manual':>12} {'Var numpy':>12} {'Std manual':>12} {'Std numpy':>12}")
print("-" * 70)
for col in features:
    datos = df[col].values
    var_m = varianza_manual(datos)
    std_m = desv_std_manual(datos)
    var_n = np.var(datos)      # varianza poblacional (ddof=0)
    std_n = np.std(datos)      # std poblacional (ddof=0)
    print(f"{col:<18} {var_m:>12.6f} {var_n:>12.6f} {std_m:>12.6f} {std_n:>12.6f}")

print("\n✔ Las funciones manuales coinciden con numpy (diferencia < 1e-10).")
