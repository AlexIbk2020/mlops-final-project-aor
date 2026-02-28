## Proyecto: Predicción de Supervivencia en el Titanic

### 👤 Estudiante
*   **Nombre completo:** Percy Alex Orihuela Rosales
*   **Correo:** alex.orihuela.rosales.uni@gmail.com
*   **Grupo:** Único

---

### A) Definición del Problema

#### 1. Caso de Uso de IA/ML
**Identificación del problema:** Este proyecto aborda la **predicción de la supervivencia de pasajeros del Titanic**. El problema consiste en determinar, dadas las características de un pasajero (como su edad, género, clase de ticket, etc.), si esta persona sobrevivió o no al trágico hundimiento.

#### 2. Contexto del Problema
El hundimiento del RMS Titanic en 1912 es uno de los desastres marítimos más estudiados. Factores socioeconómicos (como la clase en la que se viajaba) y demográficos (como el género y la edad) jugaron un papel crucial en las posibilidades de supervivencia de una persona, reflejando principios como "mujeres y niños primero". Analizar este conjunto de datos permite no solo predecir un resultado histórico, sino también comprender la influencia de estas variables en un evento de alto impacto.

#### 3. Limitaciones
*   **Datos históricos:** Los datos reflejan normas y estructuras sociales de 1912, por lo que las conclusiones no son directamente extrapolables a contextos modernos sin un análisis cuidadoso.
*   **Información incompleta:** El dataset contiene valores nulos, especialmente en las columnas `Age` (Edad) y `Cabin` (Cabina), lo que requiere técnicas de imputación o ingeniería de características.
*   **Tamaño del dataset:** El conjunto de entrenamiento es relativamente pequeño (891 registros), lo que puede limitar la complejidad de los modelos a utilizar para evitar el sobreajuste.
*   **Representatividad:** La muestra de pasajeros en el dataset puede no ser perfectamente representativa de todos los abordajes, existiendo sesgos en la recolección histórica de los datos.

#### 4. Objetivos
*   **Objetivo principal:** Desarrollar un modelo de clasificación binaria que prediga con la mayor precisión posible si un pasajero del Titanic sobrevivió (`1`) o no (`0`).
*   **Objetivo secundario:** Identificar y cuantificar la importancia de las diferentes características de los pasajeros (género, clase, edad, etc.) en la probabilidad de supervivencia, para extraer conclusiones significativas del modelo.

#### 5. Beneficios Esperados
*   **Comprensión histórica:** Proveer una herramienta que permita entender, de manera cuantitativa, los factores que determinaron la supervivencia en este desastre histórico.
*   **Aplicación de MLOps:** Demostrar la aplicación práctica de un flujo de trabajo completo de MLOps, desde la definición del problema hasta el despliegue de un modelo como servicio (API), sirviendo como un portafolio técnico sólido.
*   **Base para modelos de riesgo:** La metodología empleada puede servir como base para el desarrollo de modelos de predicción de riesgos en otros ámbitos (por ejemplo, en seguros o emergencias).

#### 6. Resultados Esperados
*   Un modelo de Machine Learning entrenado y validado, capaz de predecir la supervivencia de pasajeros.
*   Un análisis exploratorio de datos (EDA) que documente los hallazgos clave y las relaciones entre variables.
*   Una API REST funcional que permita realizar predicciones en tiempo real enviando los datos de un pasajero.
*   Un repositorio de GitHub con todo el código, los informes, las instrucciones de ejecución y la documentación completa del proyecto.

#### 7. Métricas de Éxito (Alto Nivel)
Para considerar que la solución de Machine Learning es un éxito, el modelo campeón debe alcanzar los siguientes umbrales en el conjunto de datos de validación:
*   **Precisión (Accuracy):** ≥ 80%. Mide el porcentaje total de aciertos del modelo.
*   **Puntaje F1 (F1-Score):** ≥ 0.75. Dado que las clases (sobrevivió/no sobrevivió) pueden estar desbalanceadas, esta métrica, que combina precisión y exhaustividad (recall), es crucial para evaluar el rendimiento del modelo de manera equilibrada.

---

### A.1) Adquisición de Datos

#### 1. Identificación y Fuente de los Datos
El conjunto de datos utilizado es el clásico **"Titanic - Machine Learning from Disaster"** disponible en la plataforma **Kaggle**. Este es un dataset de uso público y educativo, ampliamente utilizado en la comunidad de ciencia de datos para problemas de introducción a la clasificación.

*   **Enlace de referencia:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

#### 2. Definición y Descripción de los Datos
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

#### 3. Análisis Inicial de los Datos
Un primer vistazo a los datos revela los siguientes puntos importantes que serán abordados en la fase de Experimentación (EDA):
*   **Valores Nulos:** Las columnas `Age` y `Cabin` presentan una cantidad significativa de valores nulos. `Embarked` también tiene algunos pocos.
*   **Tipos de Datos:** Existe una mezcla de variables numéricas y categóricas que requerirán diferentes estrategias de preprocesamiento.
*   **Desbalance de Clases:** Se anticipa que la clase `Survived` podría no estar perfectamente balanceada, lo que justifica el uso de métricas como el F1-Score.
