import os

import time

import logging

import numpy as np

import joblib

import psutil

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from typing import List

from tensorflow.keras.models import load_model

from prometheus_client import Counter, Histogram, Gauge, generate_latest

from starlette.responses import Response



# ============================================

# Configuracao de caminhos

# ============================================

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_lstm_petr4.keras')

SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')



# ============================================

# Configuracao de logging

# ============================================

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s | %(levelname)s | %(message)s',

    handlers=[

        logging.FileHandler('api_monitor.log'),

        logging.StreamHandler()

    ]

)

logger = logging.getLogger(__name__)



# ============================================

# Metricas Prometheus

# ============================================

PREDICTION_COUNT = Counter('predictions_total', 'Total de predicoes realizadas')

PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latencia das predicoes')

PREDICTION_VALUE = Gauge('prediction_value', 'Ultimo valor predito')

MEMORY_USAGE = Gauge('memory_usage_percent', 'Uso de memoria RAM (%)')

CPU_USAGE = Gauge('cpu_usage_percent', 'Uso de CPU (%)')



# ============================================

# Inicializa o app FastAPI

# ============================================

app = FastAPI(

    title="API de Previsao PETR4",

    description="API para inferencia do modelo LSTM de predicao de acoes (Fase 5 - MLOps)",

    version="1.0.0"

)



# ============================================

# Schema de entrada

# ============================================

class PredictionRequest(BaseModel):

    historical_data: List[float]



# ============================================

# Variaveis globais

# ============================================

model = None

scaler = None



# ============================================

# Eventos da API

# ============================================

@app.on_event("startup")

def load_artifacts():

    """Carrega o modelo e o scaler quando a API e iniciada."""

    global model, scaler

    try:

        model = load_model(MODEL_PATH)

        scaler = joblib.load(SCALER_PATH)

        logger.info("Modelo e Scaler carregados com sucesso!")

    except Exception as e:

        logger.error(f"Erro ao carregar artefatos: {e}")



# ============================================

# Endpoints

# ============================================

@app.get("/")

def read_root():

    return {"message": "API de Previsao PETR4 ativa! Acesse /docs para testar os endpoints."}



@app.get("/health")

def health_check():

    """Endpoint de saude da API para monitoramento."""

    return {

        "status": "healthy",

        "model_loaded": model is not None,

        "scaler_loaded": scaler is not None,

        "memory_usage_percent": psutil.virtual_memory().percent,

        "cpu_usage_percent": psutil.cpu_percent(interval=0.1)

    }



@app.get("/metrics")

def metrics():

    """Endpoint que expoe metricas no formato Prometheus."""

    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))

    return Response(content=generate_latest(), media_type="text/plain")



@app.post("/predict")

def predict(request: PredictionRequest):

    """

    Recebe 60 dias de dados historicos de fechamento e preve o 61o dia.

    """

    start_time = time.time()

    data = request.historical_data



    if len(data) != 60:

        logger.warning(f"Requisicao invalida: {len(data)} dias enviados (esperado: 60)")

        raise HTTPException(

            status_code=400,

            detail=f"O modelo requer exatamente 60 dias de dados. Voce enviou {len(data)}."

        )



    # Pre-processamento

    input_data = np.array(data).reshape(-1, 1)

    input_scaled = scaler.transform(input_data)

    input_model = input_scaled.reshape(1, 60, 1)



    # Inferencia

    prediction_scaled = model.predict(input_model, verbose=0)

    prediction = scaler.inverse_transform(prediction_scaled)

    predicted_value = round(float(prediction[0][0]), 2)



    # Metricas e logging

    latency = time.time() - start_time

    PREDICTION_COUNT.inc()

    PREDICTION_LATENCY.observe(latency)

    PREDICTION_VALUE.set(predicted_value)



    logger.info(

        f"Predicao realizada | "

        f"Valor: R$ {predicted_value} | "

        f"Latencia: {latency:.4f}s | "

        f"Input (ultimo valor): {data[-1]}"

    )



    return {

        "predicted_price": predicted_value,

        "currency": "BRL",

        "symbol": "PETR4.SA"

    } 

# ============================================
# Execucao do Servidor (Configuracao Render)
# ============================================
if __name__ == "__main__":
    import uvicorn
    # Pega a porta injetada pelo Render ou usa a 8000 como padrão local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)