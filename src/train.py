import os
import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Configurações de caminhos
DATA_DIR = 'data'
MODEL_DIR = 'models'

def load_data():
    """Carrega os dados processados pelo data_prep.py"""
    print("Carregando dados processados...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    actual_prices = np.load(os.path.join(DATA_DIR, 'actual_prices.npy'))
    return X_train, y_train, X_test, actual_prices

def build_model(input_shape):
    """Constrói a arquitetura LSTM da Fase 4 com a sintaxe atualizada do Keras"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_metrics(actual_prices, predictions):
    """Calcula MAE, RMSE e MAPE"""
    mae = mean_absolute_error(actual_prices, predictions)
    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
    return mae, rmse, mape

def main():
    # 1. Configurar o MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # Salva os logs em um arquivo local
    mlflow.set_experiment("Previsao_PETR4_LSTM")

    with mlflow.start_run():
        X_train, y_train, X_test, actual_prices = load_data()
        
        # 2. Registrar hiperparâmetros no MLflow
        epochs = 15
        batch_size = 32
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("prediction_days", 60)

        # 3. Treinar o modelo
        print("Iniciando treinamento do modelo...")
        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # 4. Fazer previsões
        print("Realizando previsões para avaliação...")
        predictions_scaled = model.predict(X_test)
        
        # Carregar o scaler para reverter a normalização
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        predictions = scaler.inverse_transform(predictions_scaled)

        # Como pegamos 60 dias de janela para o teste, o tamanho das previsões será igual
        # ao tamanho do actual_prices, mas vamos garantir que tenham o mesmo shape
        min_len = min(len(actual_prices), len(predictions))
        actual_prices = actual_prices[-min_len:]
        predictions = predictions[-min_len:]

        # 5. Calcular e registrar métricas
        mae, rmse, mape = calculate_metrics(actual_prices, predictions)
        print(f"\nMétricas Finais -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        # 6. Salvar o modelo (formato moderno .keras e no MLflow)
        model_path = os.path.join(MODEL_DIR, 'modelo_lstm_petr4.keras')
        model.save(model_path)
        mlflow.keras.log_model(model, "modelo_lstm")
        print(f"Modelo salvo em {model_path} e registrado no MLflow.")

if __name__ == "__main__":
    main()