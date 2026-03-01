"""
src/data_preparation.py
Preparación de datos para Titanic - Versión final para entrenamiento
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_raw_data(data_path='data/raw/'):
    """Cargar datos crudos"""
    print("📂 Cargando datos crudos...")
    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def clean_titanic_data(df, is_train=True):
    """Limpiar datos (misma lógica del notebook)"""
    df_clean = df.copy()
    
    # Age
    if 'Age' in df_clean.columns:
        df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    
    # Embarked (solo en train)
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Fare (solo en test puede tener NaN)
    if 'Fare' in df_clean.columns:
        df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    # Cabin -> feature binaria
    if 'Cabin' in df_clean.columns:
        df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
        df_clean.drop('Cabin', axis=1, inplace=True)
    
    # Eliminar columnas no útiles
    cols_to_drop = ['Name', 'Ticket', 'PassengerId']
    df_clean.drop([c for c in cols_to_drop if c in df_clean.columns], 
                  axis=1, inplace=True)
    
    return df_clean

def feature_engineering(df):
    """Crear nuevas características"""
    df_fe = df.copy()
    
    if 'SibSp' in df_fe.columns and 'Parch' in df_fe.columns:
        df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
        df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    
    return df_fe

def prepare_training_data():
    """Función principal que orquesta toda la preparación"""
    
    print("="*60)
    print("🚢 PREPARACIÓN DE DATOS TITANIC - VERSIÓN ENTRENAMIENTO")
    print("="*60)
    
    # 1. Cargar datos crudos
    train_raw, test_raw = load_raw_data()
    
    # 2. Separar target
    y_train = train_raw['Survived'].copy()
    train_features = train_raw.drop('Survived', axis=1)
    
    # 3. Limpiar y hacer feature engineering
    print("\n🧹 Limpiando y creando características...")
    train_clean = clean_titanic_data(train_features)
    test_clean = clean_titanic_data(test_raw, is_train=False)
    
    train_fe = feature_engineering(train_clean)
    test_fe = feature_engineering(test_clean)
    
    # 4. Codificar variables categóricas
    print("\n🔄 Codificando variables categóricas...")
    encoders = {}
    categorical_cols = train_fe.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        train_fe[col] = encoders[col].fit_transform(train_fe[col])
        # Aplicar mismo encoder a test
        test_fe[col] = test_fe[col].map(lambda x: -1 if x not in encoders[col].classes_ 
                                         else encoders[col].transform([x])[0])
        test_fe[col].fillna(-1, inplace=True)
    
    # 5. Escalar características numéricas
    print("\n📏 Escalando características numéricas...")
    scaler = StandardScaler()
    numerical_cols = train_fe.select_dtypes(include=['int64', 'float64']).columns
    
    X_train_scaled = train_fe.copy()
    X_test_scaled = test_fe.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(train_fe[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(test_fe[numerical_cols])
    
    # 6. DESCRIPCIÓN DE CARACTERÍSTICAS (Requisito de la Parte D)
    print("\n📊 DESCRIPCIÓN DE CARACTERÍSTICAS FINALES:")
    print("="*60)
    feature_desc = {}
    for col in X_train_scaled.columns:
        dtype = X_train_scaled[col].dtype
        desc = {
            'tipo': 'numérico' if dtype in ['int64', 'float64'] else 'categórico',
            'valores_unicos': int(X_train_scaled[col].nunique()),
            'rango': f"{X_train_scaled[col].min():.2f} a {X_train_scaled[col].max():.2f}",
            'media': float(X_train_scaled[col].mean()),
            'std': float(X_train_scaled[col].std())
        }
        feature_desc[col] = desc
        print(f"\n🔹 {col}:")
        print(f"   - Tipo: {desc['tipo']}")
        print(f"   - Valores únicos: {desc['valores_unicos']}")
        print(f"   - Rango: {desc['rango']}")
        if desc['tipo'] == 'numérico':
            print(f"   - Media: {desc['media']:.4f}")
            print(f"   - Std: {desc['std']:.4f}")
    
    # 7. Guardar datasets finales
    print("\n💾 Guardando datasets finales...")
    os.makedirs('data/training', exist_ok=True)
    
    X_train_scaled.to_csv('data/training/X_train.csv', index=False)
    X_test_scaled.to_csv('data/training/X_test.csv', index=False)
    pd.DataFrame({'Survived': y_train}).to_csv('data/training/y_train.csv', index=False)
    
    # Guardar preprocesadores para usarlos en entrenamiento
    joblib.dump(encoders, 'data/training/encoders.pkl')
    joblib.dump(scaler, 'data/training/scaler.pkl')
    
    # Guardar descripción de características como JSON
    with open('data/training/feature_description.json', 'w') as f:
        json.dump(feature_desc, f, indent=2)
    
    print("\n✅ DATOS LISTOS PARA ENTRENAMIENTO:")
    print(f"   X_train: {X_train_scaled.shape}")
    print(f"   X_test: {X_test_scaled.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"\n   Archivos guardados en: data/training/")
    print(f"   - X_train.csv, X_test.csv, y_train.csv")
    print(f"   - encoders.pkl, scaler.pkl")
    print(f"   - feature_description.json")
    
    return X_train_scaled, X_test_scaled, y_train

if __name__ == "__main__":
    X_train, X_test, y_train = prepare_training_data()