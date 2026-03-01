"""
src/schema.py
Esquemas de validación para la API del modelo Titanic
Versión compatible con Pydantic v2
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict

class TitanicInput(BaseModel):
    """Esquema de entrada para predicciones"""
    
    Pclass: int = Field(..., ge=1, le=3, description="Clase del pasajero (1, 2, 3)")
    Sex: Literal['male', 'female'] = Field(..., description="Género")
    Age: float = Field(..., ge=0, le=100, description="Edad en años")
    SibSp: int = Field(..., ge=0, le=10, description="Número de hermanos/cónyuges")
    Parch: int = Field(..., ge=0, le=10, description="Número de padres/hijos")
    Fare: float = Field(..., ge=0, le=600, description="Tarifa del pasaje")
    Embarked: Literal['C', 'Q', 'S'] = Field(..., description="Puerto de embarque")
    
    @field_validator('Age')
    @classmethod
    def age_must_be_reasonable(cls, v):
        """Validar que la edad sea razonable"""
        if v < 0 or v > 100:
            raise ValueError('La edad debe estar entre 0 y 100 años')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 25.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S"
            }
        }
    }

class TitanicOutput(BaseModel):
    """Esquema de salida de predicciones"""
    
    prediction: int = Field(..., description="Predicción (0 = No sobrevivió, 1 = Sobrevivió)")
    survival_probability: float = Field(..., ge=0, le=1, description="Probabilidad de supervivencia")
    survival_prediction: str = Field(..., description="Interpretación ('Sobrevivió' o 'No sobrevivió')")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilidades por clase")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "survival_probability": 0.189,
                "survival_prediction": "No sobrevivió",
                "class_probabilities": {
                    "no_survive": 0.811,
                    "survive": 0.189
                }
            }
        }
    }