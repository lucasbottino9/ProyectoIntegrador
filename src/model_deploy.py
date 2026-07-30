"""
===========================================================
Proyecto Integrador - MLOps
Henry

Archivo:
    model_deploy.py

Descripción:
    API para desplegar el modelo de Machine Learning.

    Funcionalidades:
        - Carga del modelo entrenado
        - Predicción individual
        - Predicción batch
        - Endpoint REST mediante FastAPI
Versión:
    1.2.0
===========================================================
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# CONFIGURACIÓN
BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    BASE_PATH /
    "models" /
    "best_model.pkl"
)

# CARGA DEL MODELO
def load_model():
    """
    Carga el modelo entrenado.
    El archivo debe contener
    el pipeline completo:
        preprocessing
              +
        modelo
    """
    try:
        model = joblib.load(
            MODEL_PATH
        )
        print(
            "Modelo cargado correctamente"
        )
        return model
    except Exception as error:
        print(
            "Error cargando modelo:",
            error
        )
        return None
model = load_model()

# CREACIÓN DE API
app = FastAPI(
    title="Credit Payment Prediction API",
    description="""
    API para predicción de pago
    oportuno de créditos.
    Modelo desarrollado como parte
    del Proyecto Integrador MLOps.
    """,
    version="1.0.0"
)

# MODELOS DE ENTRADA
class CreditData(BaseModel):
    """
    Define un registro individual.
    Los campos corresponden
    al dataset original.
    """
    tipo_credito: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str
    
# FUNCIÓN DE PREDICCIÓN
def make_prediction(
        data: pd.DataFrame
):
    """
    Ejecuta predicción utilizando
    el pipeline cargado.
    """
    if model is None:
        raise Exception(
            "Modelo no disponible"
        )
    prediction = model.predict(
        data
    )
    probability = None
    if hasattr(
        model,
        "predict_proba"
    ):
        probability = (
            model.predict_proba(data)
            [:,1]
            .tolist(
        )
    return prediction.tolist(), probability

# ENDPOINT HOME
@app.get("/")
def home():
    return {
        "mensaje":
        "API de predicción crediticia funcionando",
        "version":
        "1.0.0"
    }

# ENDPOINT PREDICCIÓN INDIVIDUAL
@app.post("/predict")
def predict(
        credit: CreditData
):
    try:
        data = pd.DataFrame(
            [
                credit.model_dump()
            ]
        )
        prediction, probability = make_prediction(
            data
        )
        return {
            "prediction":
                prediction[0],
            "probability":
                probability[0]
                if probability
                else None
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

# ENDPOINT PREDICCIÓN BATCH
@app.post("/predict_batch")
def predict_batch(
        credits: List[CreditData]
):
    try:
        data = pd.DataFrame(
            [
                credit.model_dump()
                for credit in credits
            ]
        )
        predictions, probabilities = make_prediction(
            data
        )
        results=[]
        for index, prediction in enumerate(predictions):
            results.append(
                {
                "prediction":
                    prediction,
                "probability":
                    probabilities[index]
                    if probabilities
                    else None
                }
            )
        return {
            "total_registros":
                len(results),
            "predicciones":
                results
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

# EJECUCIÓN LOCAL
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )