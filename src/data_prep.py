import os
import yfinance as yf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Configurações iniciais
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
PREDICTION_DAYS = 60
MODEL_DIR = 'models'
DATA_DIR = 'data'

def create_directories():
    """Garante que os diretórios de saída existam."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def download_data(symbol, start, end):
    """Baixa os dados do Yahoo Finance."""
    print(f"Baixando dados para {symbol}...")
    df = yf.download(symbol, start=start, end=end, timeout=20)
    # Pega apenas a coluna Close (ajuste para a estrutura multi-index do yfinance recente)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[['Close']].values

def prepare_data(dataset):
    """Normaliza e separa em treino e teste com janelas móveis."""
    print("Preparando e normalizando os dados...")
    train_size = int(len(dataset) * 0.8)
    train_data_raw = dataset[:train_size]
    test_data_raw = dataset[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data_raw)
    test_scaled = scaler.transform(test_data_raw)

    # Salvando o scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Scaler salvo com sucesso.")

    # Função auxiliar para criar janelas
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, PREDICTION_DAYS)
    

    inputs_test = np.concatenate((train_scaled[-PREDICTION_DAYS:], test_scaled), axis=0)
    X_test, y_test = create_sequences(inputs_test, PREDICTION_DAYS)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, test_data_raw

def main():
    create_directories()
    dataset = download_data(SYMBOL, START_DATE, END_DATE)
    X_train, y_train, X_test, y_test, test_data_raw = prepare_data(dataset)
    
    # Salvando os arrays processados para o script de treino usar
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(DATA_DIR, 'actual_prices.npy'), test_data_raw) 

    print(f"Dados salvos em {DATA_DIR}. Treino shape: {X_train.shape}")

if __name__ == "__main__":
    import pandas as pd 
    main()