"""
src/mlflow_register.py
Registro del modelo en MLflow (opcional)
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import glob
import os

def register_model_with_mlflow(model_path, experiment_name="Titanic"):
    """Registrar modelo en MLflow"""
    
    print(f"\n Procesando modelo: {model_path}")
    
    # Cargar modelo
    modelo = joblib.load(model_path)
    
    # Cargar datos de validación
    X_train = pd.read_csv('data/training/X_train.csv')
    y_train = pd.read_csv('data/training/y_train.csv').values.ravel()
    
    # Configurar MLflow
    mlflow.set_experiment(experiment_name)
    print(f" Experimento: {experiment_name}")
    
    with mlflow.start_run():
        # Logear parámetros
        mlflow.log_params(modelo.get_params())
        print("    Parámetros guardados")
        
        # Logear métricas
        y_pred = modelo.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        print(f"    Métricas guardadas: accuracy={accuracy:.4f}, f1={f1:.4f}")
        
        # Registrar modelo
        mlflow.sklearn.log_model(modelo, "model")
        print("    Modelo guardado en MLflow")
        
        # Registrar en Model Registry
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="Titanic_Survival_Model"
        )
        
        print(f"\n Modelo registrado exitosamente en MLflow")
        print(f"    Run ID: {mlflow.active_run().info.run_id}")
        print(f"    Modelo: {result.name}")
        print(f"    Versión: {result.version}")
        
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    # Buscar el modelo más reciente en models/
    model_files = glob.glob('models/*.pkl')
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(" Buscando modelo más reciente...")
        register_model_with_mlflow(latest_model)
    else:
        print(" No se encontraron modelos en models/")
        print(" Ejecuta primero: python src/train.py")