import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model

# Configuração de caminhos baseados na estrutura do projeto
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_lstm_petr4.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Inicializa o app FastAPI
app = FastAPI(
    title="API de Previsão PETR4",
    description="API para inferência do modelo LSTM de predição de ações (Fase 5 - MLOps)",
    version="1.0.0"
)

# Definição do schema de entrada usando Pydantic para validação automática
class PredictionRequest(BaseModel):
    historical_data: List[float]

# Variáveis globais para armazenar o modelo e o scaler na memória
model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    """Carrega o modelo e o scaler quando a API é iniciada."""
    global model, scaler
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelo e Scaler carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar artefatos: {e}")

@app.get("/")
def read_root():
    return {"message": "API de Previsão PETR4 ativa! Acesse /docs para testar os endpoints."}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Recebe 60 dias de dados históricos de fechamento e prevê o 61º dia.
    """
    data = request.historical_data
    
    # Validação rigorosa: nosso modelo foi treinado com janelas de 60 dias
    if len(data) != 60:
        raise HTTPException(
            status_code=400, 
            detail=f"O modelo requer exatamente 60 dias de dados. Você enviou {len(data)}."
        )
    
    # Pré-processamento igual ao feito no treino
    input_data = np.array(data).reshape(-1, 1)
    input_scaled = scaler.transform(input_data)
    
    # Reshape para [samples, time_steps, features] que o LSTM exige
    input_model = input_scaled.reshape(1, 60, 1)
    
    # Inferência
    prediction_scaled = model.predict(input_model)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return {
        "predicted_price": round(float(prediction[0][0]), 2),
        "currency": "BRL",
        "symbol": "PETR4.SA"
    }