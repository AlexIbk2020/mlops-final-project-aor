### **Opción 2: Crear un script que genere un reporte (`src/analyze_features.py`)**

```python
"""
src/analyze_features.py
Análisis y justificación de características
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_features():
    """Analizar y documentar características finales"""
    
    print("="*60)
    print("ANÁLISIS DE CARACTERÍSTICAS - TITANIC")
    print("="*60)
    
    # Cargar datos
    X_train = pd.read_csv('data/training/X_train.csv')
    
    # 1. Descripción de cada característica
    feature_desc = {}
    print("\n DESCRIPCIÓN DE CARACTERÍSTICAS:\n")
    
    for col in X_train.columns:
        desc = {
            'tipo': str(X_train[col].dtype),
            'valores_unicos': int(X_train[col].nunique()),
            'min': float(X_train[col].min()),
            'max': float(X_train[col].max()),
            'media': float(X_train[col].mean()),
            'std': float(X_train[col].std())
        }
        feature_desc[col] = desc
        
        print(f"\n🔹 {col}:")
        print(f"   - Tipo: {desc['tipo']}")
        print(f"   - Valores únicos: {desc['valores_unicos']}")
        print(f"   - Rango: [{desc['min']:.2f}, {desc['max']:.2f}]")
        print(f"   - Media: {desc['media']:.4f}")
        print(f"   - Std: {desc['std']:.4f}")
    
    # 2. Justificación de transformaciones
    print("\n" + "="*60)
    print(" JUSTIFICACIÓN DE TRANSFORMACIONES")
    print("="*60)
    
    justificacion = {
        'Age': 'Imputación con mediana - Mantiene distribución sin sesgo',
        'Embarked': 'Imputación con moda - Solo 2 valores nulos',
        'Cabin': 'Convertida a Has_Cabin (binaria) - Indica acceso a cabina',
        'FamilySize': 'Creada como SibSp + Parch + 1 - Mide tamaño familiar',
        'IsAlone': 'Derivada de FamilySize - Identifica pasajeros solos'
    }
    
    for feature, reason in justificacion.items():
        if feature in X_train.columns:
            print(f" {feature}: {reason}")
    
    # 3. Guardar análisis
    os.makedirs('reports', exist_ok=True)
    
    with open('reports/feature_analysis.json', 'w') as f:
        json.dump({
            'descripcion_caracteristicas': feature_desc,
            'justificacion_transformaciones': justificacion
        }, f, indent=2)
    
    print(f"\n Análisis guardado en: reports/feature_analysis.json")
    
    # 4. Generar gráficos
    print("\n Generando visualizaciones...")
    
    # Distribuciones
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(X_train.columns):
        X_train[col].hist(ax=axes[i], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
    
    # Ocultar ejes vacíos
    for j in range(len(X_train.columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('reports/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Matriz de correlación
    plt.figure(figsize=(12, 10))
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación - Características Finales', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n Visualizaciones guardadas en: reports/")

if __name__ == "__main__":
    analyze_features()