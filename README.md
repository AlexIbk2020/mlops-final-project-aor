# 🚢 Proyecto: Predicción de Supervivencia en el Titanic

### 👤 Estudiante
*   **Nombre completo:** Percy Alex Orihuela Rosales
*   **Correo:** alex.orihuela.rosales.uni@gmail.com
*   **Grupo:** Maestria Data Ciencia UNI

---

## Definición del Problema

### 1. Caso de Uso de IA/ML
**Identificación del problema:** Este proyecto aborda la **predicción de la supervivencia de pasajeros del Titanic**. El problema consiste en determinar, dadas las características de un pasajero (como su edad, género, clase de ticket, etc.), si esta persona sobrevivió o no al trágico hundimiento.

### 2. Contexto del Problema
El hundimiento del RMS Titanic en 1912 es uno de los desastres marítimos más estudiados. Factores socioeconómicos (como la clase en la que se viajaba) y demográficos (como el género y la edad) jugaron un papel crucial en las posibilidades de supervivencia de una persona, reflejando principios como "mujeres y niños primero". Analizar este conjunto de datos permite no solo predecir un resultado histórico, sino también comprender la influencia de estas variables en un evento de alto impacto.

### 3. Limitaciones
*   **Datos históricos:** Los datos reflejan normas y estructuras sociales de 1912, por lo que las conclusiones no son directamente extrapolables a contextos modernos sin un análisis cuidadoso.
*   **Información incompleta:** El dataset contiene valores nulos, especialmente en las columnas `Age` (Edad) y `Cabin` (Cabina), lo que requiere técnicas de imputación o ingeniería de características.
*   **Tamaño del dataset:** El conjunto de entrenamiento es relativamente pequeño (891 registros), lo que puede limitar la complejidad de los modelos a utilizar para evitar el sobreajuste.
*   **Representatividad:** La muestra de pasajeros en el dataset puede no ser perfectamente representativa de todos los abordajes, existiendo sesgos en la recolección histórica de los datos.

### 4. Objetivos
*   **Objetivo principal:** Desarrollar un modelo de clasificación binaria que prediga con la mayor precisión posible si un pasajero del Titanic sobrevivió (`1`) o no (`0`).
*   **Objetivo secundario:** Identificar y cuantificar la importancia de las diferentes características de los pasajeros (género, clase, edad, etc.) en la probabilidad de supervivencia, para extraer conclusiones significativas del modelo.

### 5. Beneficios Esperados
*   **Comprensión histórica:** Proveer una herramienta que permita entender, de manera cuantitativa, los factores que determinaron la supervivencia en este desastre histórico.
*   **Aplicación de MLOps:** Demostrar la aplicación práctica de un flujo de trabajo completo de MLOps, desde la definición del problema hasta el despliegue de un modelo como servicio (API), sirviendo como un portafolio técnico sólido.
*   **Base para modelos de riesgo:** La metodología empleada puede servir como base para el desarrollo de modelos de predicción de riesgos en otros ámbitos (por ejemplo, en seguros o emergencias).

### 6. Resultados Esperados
*   Un modelo de Machine Learning entrenado y validado, capaz de predecir la supervivencia de pasajeros.
*   Un análisis exploratorio de datos (EDA) que documente los hallazgos clave y las relaciones entre variables.
*   Una API REST funcional que permita realizar predicciones en tiempo real enviando los datos de un pasajero.
*   Un repositorio de GitHub con todo el código, los informes, las instrucciones de ejecución y la documentación completa del proyecto.

### 7. Métricas de Éxito (Alto Nivel)
Para considerar que la solución de Machine Learning es un éxito, el modelo campeón debe alcanzar los siguientes umbrales en el conjunto de datos de validación:
*   **Precisión (Accuracy):** ≥ 80%. Mide el porcentaje total de aciertos del modelo.
*   **Puntaje F1 (F1-Score):** ≥ 0.75. Dado que las clases (sobrevivió/no sobrevivió) pueden estar desbalanceadas, esta métrica, que combina precisión y exhaustividad (recall), es crucial para evaluar el rendimiento del modelo de manera equilibrada.

---


### DESARROLLO
## A.1) Adquisición de Datos

### 1. Identificación y Fuente de los Datos
El conjunto de datos utilizado es el clásico **"Titanic - Machine Learning from Disaster"** disponible en la plataforma **Kaggle**. Este es un dataset de uso público y educativo, ampliamente utilizado en la comunidad de ciencia de datos para problemas de introducción a la clasificación.

*   **Enlace de referencia:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

### 2. Definición y Descripción de los Datos
Los datos se componen de tres archivos, ubicados en la carpeta `data/raw/` del repositorio:
*   **`train.csv` (Datos de Entrenamiento):** Contiene 891 registros de pasajeros. Incluye la variable objetivo `Survived`, que indica si el pasajero sobrevivió (1) o no (0). Este archivo se utiliza para entrenar y validar los modelos.
*   **`test.csv` (Datos de Prueba):** Contiene 418 registros de pasajeros. No incluye la columna `Survived`. Se utiliza para realizar predicciones finales con el modelo entrenado.
*   **`gender_submission.csv` (Ejemplo de Formato):** Es un archivo de ejemplo proporcionado por Kaggle que muestra el formato esperado para las predicciones finales (solo las columnas `PassengerId` y `Survived`). Sirve como referencia.

A continuación, se describen las variables presentes en el conjunto de datos:

| Variable | Descripción | Tipo | Rango / Valores Posibles |
| :--- | :--- | :--- | :--- |
| **PassengerId** | Identificador único del pasajero. | Entero | 1 a 1309 |
| **Survived** | **Variable objetivo:** Indica si el pasajero sobrevivió. | Binaria | 0 = No, 1 = Sí |
| **Pclass** | Clase del ticket (como proxy del estatus socioeconómico). | Categórica Ordinal | 1 = Primera, 2 = Segunda, 3 = Tercera |
| **Name** | Nombre completo del pasajero. | Texto | - |
| **Sex** | Género del pasajero. | Categórica | male, female |
| **Age** | Edad del pasajero en años. | Numérica Continua | 0.42 a 80.0 (con valores nulos) |
| **SibSp** | Número de hermanos/esposos (spouses) a bordo. | Entera | 0 a 8 |
| **Parch** | Número de padres/hijos a bordo. | Entera | 0 a 9 |
| **Ticket** | Número del ticket. | Texto | - |
| **Fare** | Tarifa del pasajero. | Numérica Continua | 0 a 512.33 |
| **Cabin** | Número de la cabina. | Texto | (muchos valores nulos) |
| **Embarked** | Puerto de embarque. | Categórica | C = Cherburgo, Q = Queenstown, S = Southampton |

### 3. Análisis Inicial de los Datos
Un primer vistazo a los datos revela los siguientes puntos importantes que serán abordados en la fase de Experimentación (EDA):
*   **Valores Nulos:** Las columnas `Age` y `Cabin` presentan una cantidad significativa de valores nulos. `Embarked` también tiene algunos pocos.
*   **Tipos de Datos:** Existe una mezcla de variables numéricas y categóricas que requerirán diferentes estrategias de preprocesamiento.
*   **Desbalance de Clases:** Se anticipa que la clase `Survived` podría no estar perfectamente balanceada, lo que justifica el uso de métricas como el F1-Score.

### Para el análisis de datos se creo un notebook 03_feature_analysis.ipynb
Aquí podrá encontrar las siguientes características:
## 📓 Análisis del Notebook: `03_feature_analysis.ipynb`

A continuación, se detalla el contenido y los análisis realizados en el notebook de justificación de características:

### **Celda 2 - Cargar Datos Originales**
- **Carga de datos:** `train.csv` y `test.csv` desde `../data/raw/`
- **Dimensiones verificadas:** Train (891, 12), Test (418, 11)
- **Visualización:** Primeras filas de los datasets

### **Celda 3 - Información del Dataset**
- **Tipos de datos:** Uso de `info()` para identificar tipos
- **Estadísticas:** `describe()` para análisis descriptivo
- **Valores nulos:** Identificación inicial de columnas con nulos

### **Celda 4 - Análisis de Valores Nulos**
- **Conteo por columna:** Identificación precisa de valores nulos
- **Porcentajes:** Cálculo de porcentajes sobre el total
- **Visualización:** Mapa de calor (heatmap) de valores nulos
- **Resultados clave:**
  - Age: 177 nulos (19.9%)
  - Cabin: 687 nulos (77%)
  - Embarked: 2 nulos (0.2%)

### **Celda 5 - Crear Copia para Trabajar**
- **Copia:** `df = train_raw.copy()`
- **Justificación:** Preservar datos originales inalterados

### **Celda 6 - Transformación 1: Age (Imputación)**
- **Análisis:** Distribución original de edad
- **Comparación:** Media vs Mediana (visual)
- **Justificación:** Uso de mediana por ser robusta a outliers
- **Aplicación:** `fillna(age_median)`
- **Verificación:** 177 valores imputados correctamente

### **Celda 7 - Transformación 2: Embarked (Imputación)**
- **Análisis:** Distribución de puertos de embarque
- **Moda identificada:** 'S' (Southampton)
- **Justificación:** Solo 2 valores nulos (0.2%)
- **Aplicación:** `fillna(embarked_mode)`
- **Verificación:** Distribución final consistente

### **Celda 8 - Transformación 3: Fare (Preparación para Test)**
- **Verificación:** Train sin nulos en Fare
- **Detección:** 1 valor nulo en test
- **Cálculo:** Mediana de Fare en train
- **Justificación:** Imputar valor nulo de test con mediana de train

### **Celda 9 - Transformación 4: Cabin → Has_Cabin**
- **Análisis:** 77% de valores nulos en Cabin
- **Justificación:** Imposible imputar, pero presencia de cabina es informativa
- **Creación:** Variable binaria `Has_Cabin` (1 = tiene cabina, 0 = no tiene)
- **Conteo resultante:** 23% con cabina, 77% sin cabina
- **Limpieza:** Eliminación de columna original `Cabin`

### **Celda 10 - Eliminar Columnas No Útiles**
- **Columnas identificadas:** `Name`, `Ticket`, `PassengerId`
- **Justificaciones:**
  - `Name`: Demasiados valores únicos, difícil de codificar
  - `Ticket`: Formato inconsistente, poco informativo
  - `PassengerId`: Solo identificador, sin valor predictivo
- **Eliminación:** Columnas removidas del dataset
- **Verificación:** Columnas restantes listadas

### **Celda 11 - Feature Engineering: FamilySize**
- **Creación:** `FamilySize = SibSp + Parch + 1`
- **Justificación:** Medir el tamaño del grupo familiar
- **Análisis:** Distribución de tamaños familiares
- **Visualización:** Tasa de supervivencia por tamaño familiar

### **Celda 12 - Feature Engineering: IsAlone**
- **Creación:** `IsAlone = (FamilySize == 1).astype(int)`
- **Justificación:** Identificar pasajeros que viajan solos
- **Conteo:** 60% viajan solos, 40% acompañados
- **Análisis de supervivencia:** Solos (30%) vs Acompañados (50%)

### **Celda 13 - Cargar Datos Finales (Procesados)**
- **Carga:** `X_train.csv` y `y_train.csv` desde `../data/training/`
- **Dimensiones finales:** X_train (891, 10), y_train (891,)
- **Listado:** 10 características finales del modelo

### **Celda 14 - Estadísticas de Características Finales**
- **Estadísticas descriptivas:** `describe()` aplicado
- **Tipos de datos:** Verificación por columna
- **Valores únicos:** Conteo por característica

### **Celda 15 - Visualización de Distribuciones**
- **Histogramas:** Grid 3x4 con distribuciones de cada característica
- **Análisis visual:** Identificación de patrones y posibles outliers
- **Patrones detectados:** Distribuciones esperadas según la naturaleza de cada variable

### **Celda 16 - Matriz de Correlación**
- **Cálculo:** Correlaciones entre todas las características
- **Visualización:** Heatmap de correlaciones
- **Identificación:** Relaciones fuertes entre variables
- **Análisis:** Interpretación de correlaciones significativas

### **Celda 17 - Importancia de Características**
- **Modelo temporal:** Random Forest para calcular importancia
- **Feature importances:** Ranking de características
- **Top 5 más importantes:**
  1. **Sex** (género) - Impacto más significativo
  2. **Pclass** (clase) - Factor socioeconómico determinante
  3. **Age** (edad) - Influencia en supervivencia
  4. **Fare** (tarifa) - Relacionado con clase
  5. **FamilySize** (tamaño familiar) - Factor adicional
- **Visualización:** Gráfico de barras de importancias

### **Celda 18 - Conclusiones Finales**

#### **Resumen de Transformaciones Aplicadas:**
| Característica | Transformación | Justificación |
|----------------|----------------|---------------|
| Age | Imputación con mediana | 177 nulos, robustez ante outliers |
| Embarked | Imputación con moda | Solo 2 nulos, puerto más común 'S' |
| Cabin | Convertida a Has_Cabin | 77% nulos, pero tener cabina es informativo |
| FamilySize | Creada (SibSp + Parch + 1) | Medir tamaño familiar |
| IsAlone | Derivada de FamilySize | Identificar pasajeros solos |
| Name, Ticket, PassengerId | Eliminadas | Sin valor predictivo |

#### **Características Finales (10 variables):**
`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Has_Cabin`, `FamilySize`, `IsAlone`

#### **Hallazgos Clave:**
- **Sex, Pclass y Age** son las 3 características más importantes
- Las transformaciones aplicadas mejoran el poder predictivo
- Tener cabina (Has_Cabin=1) correlaciona con mayor supervivencia
- Viajar solo (IsAlone=1) disminuye probabilidad de supervivencia

#### **Conclusión Final:**
Todas las transformaciones realizadas están debidamente justificadas mediante análisis visual y estadístico, preparando óptimamente los datos para la fase de modelado.

---

## 📁 B) Preparación del Proyecto

*    Repositorio GitHub: [uni_mds_ciclo3_ml_project](https://github.com/tu-usuario/uni_mds_ciclo3_ml_project)
*    Rama principal: `main`
*    Rama de desarrollo: `develop`
*    Estructura de proyecto organizada:

mlops-final-project-aor/
├── data/
│ ├── raw/ # Datos crudos (train.csv, test.csv, gender_submission.csv)
│ └── training/ # Datos procesados para entrenamiento
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── scaler.pkl
│ ├── encoders.pkl
│ └── feature_description.json
├── notebooks/
│ ├── 01_exploracion.ipynb # Análisis exploratorio inicial
│ ├── 02_experimentacion.ipynb # Experimentación con modelos
│ ├── 03_feature_analysis.ipynb # Análisis y justificación de características
│ └── 04_model_serving.ipynb # Implementación y prueba de la API
├── src/
│ ├── data_preparation.py # Script de preparación de datos
│ ├── train.py # Script de entrenamiento
│ ├── schema.py # Esquemas Pydantic para la API
│ └── serving.py # API REST con FastAPI
├── models/ # Modelos entrenados (.pkl)
├── experiments/ # Métricas de experimentos (.json)
├── reports/ # Reportes y gráficos generados
│ ├── comparacion_modelos_detallada.png
│ ├── matriz_confusion_campeon.png
│ ├── curva_roc_campeon.png
│ ├── importancia_features_campeon.png
│ ├── api_test_.json
│ └── submission_.csv
├── api.log # Logs de la API
└── README.md # Este archivo


---

## C) Experimentación de ML

### Transformaciones Aplicadas
- Imputación de valores nulos (Age con mediana, Embarked con moda).
- Feature engineering: FamilySize, IsAlone, Has_Cabin.
- Codificación de variables categóricas (Sex, Embarked).
- Escalado de características numéricas.

### Modelos Evaluados
| Modelo | Accuracy | F1-Score | ROC-AUC |
|--------|----------|-----------|---------|
| **Regresión Logística** | **81.56%** | **75.56%** | **85.53%** |
| SVM | 81.01% | 73.44% | 83.85% |
| Gradient Boosting | 80.45% | 71.54% | 81.36% |
| KNN | 77.65% | 71.43% | 82.76% |
| Random Forest | 77.09% | 69.63% | 82.73% |
| Árbol de Decisión | 73.18% | 64.18% | 70.26% |

### Modelo Campeón
**Regresión Logística** - Cumple métricas de éxito:
-  Accuracy: **81.56%** (≥80%)
-  F1-Score: **75.56%** (≥0.75)
-  ROC-AUC: **85.53%**

 **Resultados completos en:** `notebooks/02_experimentacion.ipynb`

---

##  D) Actividades de Desarrollo de ML

### Preparación de Datos
-  Datos crudos en `data/raw/`
-  Script modular: `src/data_preparation.py`
-  Dataset final en `data/training/`

### Descripción de Características Finales
| Característica | Tipo | Descripción |
|----------------|------|-------------|
| Pclass | int | Clase del pasajero (1, 2, 3) |
| Sex | int | Género codificado (0=male, 1=female) |
| Age | float | Edad imputada y escalada |
| SibSp | int | # hermanos/cónyuges |
| Parch | int | # padres/hijos |
| Fare | float | Tarifa escalada |
| Embarked | int | Puerto codificado |
| Has_Cabin | int | ¿Tiene cabina? (0/1) |
| FamilySize | int | Tamaño familiar |
| IsAlone | int | ¿Viaja solo? (0/1) |

### Justificación de Transformaciones
| Característica | Transformación | Justificación |
|----------------|----------------|---------------|
| Age | Imputación con mediana | 177 valores nulos (19.9%), mantener distribución |
| Embarked | Imputación con moda | Solo 2 valores nulos, puerto más común 'S' |
| Cabin | Convertida a Has_Cabin | 77% nulos, pero tener cabina es informativo |
| FamilySize | SibSp + Parch + 1 | Medir tamaño familiar |
| IsAlone | FamilySize == 1 | Identificar pasajeros solos |

### Entrenamiento
-  Script: `src/train.py`
-  Validación cruzada (5 folds)
-  Modelo serializado: `models/logistic_regression_*.pkl`
-  Métricas guardadas: `experiments/metrics_*.json`

---

## 🌐 E) Implementación y Servicio del Modelo

### API REST con FastAPI
-  Archivo: `src/serving.py`
-  Esquemas: `src/schema.py`
-  Documentación interactiva: Swagger UI en `/docs`

### Endpoints Disponibles
| Endpoint | Método | Descripción | Ejemplo Respuesta |
|----------|--------|-------------|-------------------|
| `/` | GET | Health check básico | `{"message": "Titanic Survival Prediction API", "status": "online"}` |
| `/health` | GET | Estado detallado | `{"status": "healthy", "model_loaded": true}` |
| `/predict` | POST | Realizar predicción | Ver ejemplo abajo |
| `/info` | GET | Información del modelo | `{"model_type": "LogisticRegression", "features": [...]}` |

### Formato de Entrada
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 25.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

### Formato de salida
```json
{
  "prediction": 0,
  "survival_probability": 0.1337,
  "survival_prediction": "No sobrevivió",
  "class_probabilities": {
    "no_survive": 0.8663,
    "survive": 0.1337
  }
}

###  Ejemplo de uso
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"Pclass": 3, "Sex": "male", "Age": 25, 
          "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
)
print(response.json())