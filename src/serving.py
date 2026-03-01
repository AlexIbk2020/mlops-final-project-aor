"""
src/serving.py
API REST para servir el modelo Titanic usando FastAPI
Versión corregida y optimizada
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Configurar logging ANTES de cualquier otra cosa
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Intentar importar schema (debe estar en el mismo directorio)
try:
    # Primero intentar importación local (misma carpeta)
    from schema import TitanicInput, TitanicOutput
    print("✅ Schema importado localmente")
except ImportError:
    try:
        # Luego intentar importación absoluta
        from src.schema import TitanicInput, TitanicOutput
        print("✅ Schema importado desde src")
    except ImportError as e:
        print(f"⚠️ No se pudo importar schema: {e}")
        print("   Verifica que schema.py existe en src/")
        # Definir clases dummy para evitar errores de compilación
        from pydantic import BaseModel
        class TitanicInput(BaseModel): 
            class Config: pass
        class TitanicOutput(BaseModel): 
            class Config: pass

# Inicializar app
app = FastAPI(
    title="🚢 Titanic Survival Prediction API",
    description="API para predecir supervivencia en el Titanic",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
scaler = None
encoders = None

def load_artifacts():
    """Cargar modelo y preprocesadores"""
    global model, scaler, encoders
    
    try:
        print("\n🔍 Buscando modelos...")
        
        # Buscar modelo
        model_files = glob.glob('models/*.pkl')
        if not model_files:
            model_files = glob.glob('../models/*.pkl')
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            model = joblib.load(latest_model)
            print(f"✅ Modelo cargado: {os.path.basename(latest_model)}")
            logging.info(f"Modelo cargado: {os.path.basename(latest_model)}")
        else:
            print("⚠️ No se encontró modelo")
            logging.warning("No se encontró modelo")
        
        # Buscar scaler
        scaler_paths = ['data/training/scaler.pkl', '../data/training/scaler.pkl']
        for path in scaler_paths:
            if os.path.exists(path):
                scaler = joblib.load(path)
                print(f"✅ Scaler cargado: {path}")
                logging.info(f"Scaler cargado: {path}")
                break
        
        # Buscar encoders
        encoders_paths = ['data/training/encoders.pkl', '../data/training/encoders.pkl']
        for path in encoders_paths:
            if os.path.exists(path):
                encoders = joblib.load(path)
                print(f"✅ Encoders cargados: {path}")
                logging.info(f"Encoders cargados: {path}")
                break
        
        return model is not None
        
    except Exception as e:
        logging.error(f"Error cargando artefactos: {str(e)}")
        print(f"❌ Error: {e}")
        return False

def preprocess_input(data):
    """Preprocesar entrada para el modelo"""
    try:
        # Crear DataFrame
        df = pd.DataFrame([{
            'Pclass': data.Pclass,
            'Sex': 1 if data.Sex == 'female' else 0,
            'Age': data.Age,
            'SibSp': data.SibSp,
            'Parch': data.Parch,
            'Fare': data.Fare,
            'Embarked': data.Embarked
        }])
        
        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Mapeo manual de Embarked
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        df['Embarked'] = df['Embarked'].map(embarked_map).fillna(2)
        
        # Columnas esperadas
        expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                            'Embarked', 'FamilySize', 'IsAlone']
        df = df[expected_columns]
        
        # Escalar si hay scaler
        if scaler is not None:
            try:
                df_scaled = pd.DataFrame(
                    scaler.transform(df),
                    columns=expected_columns
                )
                return df_scaled
            except Exception as e:
                print(f"⚠️ Error escalando: {e}, usando datos sin escalar")
                return df
        else:
            print("⚠️ Usando datos sin escalar (no hay scaler)")
            return df
            
    except Exception as e:
        logging.error(f"Error en preprocess_input: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Cargar artefactos al iniciar"""
    print("\n" + "="*50)
    print("🚀 Iniciando API Titanic...")
    print("="*50)
    load_artifacts()
    print("="*50)
    print("✅ API lista para usar")
    print("="*50 + "\n")

@app.get("/", tags=["Health"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "🚢 Titanic Survival Prediction API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model is not None,
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=TitanicOutput, tags=["Prediction"])
async def predict(input_data: TitanicInput):
    """Realizar predicción de supervivencia"""
    try:
        if model is None:
            # Intentar cargar una vez más
            if not load_artifacts():
                raise HTTPException(
                    status_code=503, 
                    detail="Modelo no disponible. Verifica que exista un archivo .pkl en models/"
                )
        
        # Preprocesar
        X = preprocess_input(input_data)
        
        # Verificar que X no esté vacío
        if X is None or len(X) == 0:
            raise HTTPException(status_code=400, detail="Error en preprocesamiento")
        
        # Predecir
        prediction = int(model.predict(X)[0])
        
        # Obtener probabilidades
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            prob_survive = float(probabilities[1])
            prob_no_survive = float(probabilities[0])
        else:
            prob_survive = 1.0 if prediction == 1 else 0.0
            prob_no_survive = 1.0 if prediction == 0 else 0.0
        
        # Interpretación
        survival_prediction = "✅ Sobrevivió" if prediction == 1 else "❌ No sobrevivió"
        
        # Registrar predicción
        logging.info(f"Predicción: {prediction}, Prob: {prob_survive:.3f}")
        
        return TitanicOutput(
            prediction=prediction,
            survival_probability=prob_survive,
            survival_prediction=survival_prediction,
            class_probabilities={
                "no_survive": prob_no_survive,
                "survive": prob_survive
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/info", tags=["Info"])
async def model_info():
    """Información detallada del modelo"""
    if model is None:
        return {
            "model_loaded": False,
            "message": "No hay modelo cargado"
        }
    
    info = {
        "model_loaded": True,
        "model_type": type(model).__name__,
        "features": ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                    'Embarked', 'FamilySize', 'IsAlone'],
        "classes": ["No sobrevivió", "Sobrevivió"],
        "supports_probabilities": hasattr(model, 'predict_proba')
    }
    
    # Agregar parámetros si es posible
    if hasattr(model, 'get_params'):
        try:
            params = model.get_params()
            # Solo mostrar parámetros relevantes
            relevant_params = {}
            for key in ['n_estimators', 'max_depth', 'criterion']:
                if key in params:
                    relevant_params[key] = params[key]
            if relevant_params:
                info["parameters"] = relevant_params
        except:
            pass
    
    return info

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🌐 Iniciando servidor...")
    print("📝 Documentación: http://127.0.0.1:8000/docs")
    print("📊 Health check: http://127.0.0.1:8000/health")
    print("="*50 + "\n")
    
    uvicorn.run(
        "src.serving:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )