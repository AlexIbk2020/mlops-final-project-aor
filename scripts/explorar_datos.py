import pandas as pd
import numpy as np

print("="*50)
print("EXPLORANDO DATOS PROCESADOS - TITANIC")
print("="*50)

# Cargar datos
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

print("\n X_train - Primeras 5 filas:")
print(X_train.head())

print("\n X_train - Información:")
print(X_train.info())

print("\n X_train - Estadísticas:")
print(X_train.describe())

print("\n y_train - Distribución:")
print(y_train['Survived'].value_counts())
print(f"Porcentaje: {y_train['Survived'].value_counts(normalize=True) * 100}")

print("\n Dimensiones:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")