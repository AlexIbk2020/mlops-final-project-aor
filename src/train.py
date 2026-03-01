"""
src/train.py
Entrenamiento del modelo campeón (Regresión Logística) para Titanic
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def load_training_data():
    """Cargar datos preparados"""
    print(" Cargando datos de entrenamiento...")
    X_train = pd.read_csv('data/training/X_train.csv')
    y_train = pd.read_csv('data/training/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/training/X_test.csv')
    
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}")
    return X_train, y_train, X_test

def train_model(X_train, y_train):
    """Entrenar modelo campeón con validación cruzada"""
    print("\n Entrenando modelo campeón (Regresión Logística)...")
    
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    
    # Validación cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n Validación cruzada (5 folds):")
    print(f"   Accuracy promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Entrenar con todos los datos
    modelo.fit(X_train, y_train)
    
    # Evaluar en entrenamiento (para referencia)
    y_pred = modelo.predict(X_train)
    y_proba = modelo.predict_proba(X_train)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_train, y_pred)),
        'precision': float(precision_score(y_train, y_pred)),
        'recall': float(recall_score(y_train, y_pred)),
        'f1_score': float(f1_score(y_train, y_pred)),
        'roc_auc': float(roc_auc_score(y_train, y_proba)),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    }
    
    print(f"\n Métricas en entrenamiento:")
    for k, v in metrics.items():
        if k not in ['cv_mean', 'cv_std']:
            print(f"   {k}: {v:.4f}")
    
    return modelo, metrics

def save_model_and_metrics(modelo, metrics, X_train):
    """Guardar modelo y métricas"""
    print("\n Guardando modelo y métricas...")
    
    # Crear carpetas si no existen
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    
    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/logistic_regression_{timestamp}.pkl'
    joblib.dump(modelo, model_path)
    print(f"    Modelo guardado: {model_path}")
    
    # Guardar métricas
    metrics['timestamp'] = timestamp
    metrics['modelo'] = 'Regresión Logística'
    metrics['features'] = list(X_train.columns)
    
    metrics_path = f'experiments/metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Métricas guardadas: {metrics_path}")
    
    return model_path, metrics_path, timestamp  # <-- AHORA DEVUELVE timestamp

def main():
    """Función principal de entrenamiento"""
    print("="*60)
    print(" ENTRENAMIENTO MODELO TITANIC")
    print("="*60)
    
    # 1. Cargar datos
    X_train, y_train, X_test = load_training_data()
    
    # 2. Entrenar modelo
    modelo, metrics = train_model(X_train, y_train)
    
    # 3. Guardar modelo y métricas (ahora recibimos timestamp)
    model_path, metrics_path, timestamp = save_model_and_metrics(modelo, metrics, X_train)
    
    # 4. Generar predicciones para test (opcional)
    print("\n Generando predicciones para test...")
    y_test_pred = modelo.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': range(892, 892 + len(y_test_pred)),
        'Survived': y_test_pred
    })
    
    # Crear carpeta reports si no existe
    os.makedirs('reports', exist_ok=True)
    
    submission_path = f'reports/submission_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"    Submission guardada: {submission_path}")
    
    print("\n" + "="*60)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\n Resumen:")
    print(f"   - Modelo: Regresión Logística")
    print(f"   - Accuracy: {metrics['accuracy']:.2%}")
    print(f"   - F1-Score: {metrics['f1_score']:.2%}")
    print(f"   - CV Accuracy: {metrics['cv_mean']:.2%} (+/- {metrics['cv_std']:.2%})")

if __name__ == "__main__":
    main()