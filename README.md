# 🚀 Tech Challenge Fase 5 - MLOps & Engenharia de Machine Learning

Este repositório apresenta o desenvolvimento de um modelo preditivo de Deep Learning estruturado como um **produto de software automatizado, monitorado e conteinerizado** sob os princípios de **MLOps** (Machine Learning Operations). O objetivo é prever o preço de fechamento diário das ações da Petrobras (`PETR4.SA`) utilizando uma rede neural LSTM.

🔗 **Repositório GitHub:** [https://github.com/Fernandobdsantos/Tech_Challenge5](https://github.com/Fernandobdsantos/Tech_Challenge5)  
🔗 **API em Produção (Render):** [https://tech-challenge-5.onrender.com/docs](https://tech-challenge-5.onrender.com/docs)

---

## 🏗️ Arquitetura e Decisões de MLOps

Diferente de ambientes exploratórios (como Jupyter Notebooks), esta solução adota uma arquitetura modular inspirada em padrões de engenharia de software:

1. **Modularização do Pipeline:**
   * `src/data_prep.py`: Responsável pela extração automatizada de dados via Yahoo Finance, limpeza e engenharia de features (criação de janelas deslizantes de 60 dias).
   * `src/train.py`: Focado estritamente no treinamento da rede neural LSTM.

2. **Rastreamento de Experimentos (MLflow):**
   * O pipeline de treinamento está totalmente integrado ao MLflow, registrando automaticamente hiperparâmetros (épocas, tamanho de lote) e métricas de avaliação (MAE, RMSE, MAPE) a cada execução, além de versionar os artefatos de modelo (`.keras`).

3. **API de Alta Performance e Monitoramento (FastAPI & Prometheus):**
   * Desenvolvida com validação estrita de dados (`Pydantic`) para garantir o contrato de entrada (exigência exata de 60 observações históricas).
   * Carregamento otimizado: os artefatos (modelo treinado e scaler) são carregados na memória apenas na inicialização (`startup`), garantindo baixíssima latência nas respostas.
   * **Observabilidade em Produção:** Endpoints adicionais de monitoramento (`/health` para verificação de saúde e consumo de recursos, e `/metrics` para exposição de métricas no padrão Prometheus).

4. **Containerização e Orquestração (Docker & Makefile):**
   * **Docker:** Garante o isolamento completo de ambiente, permitindo que a aplicação execute de forma idêntica em qualquer sistema operacional ou servidor de nuvem.
   * **Makefile:** Orquestra todo o fluxo de trabalho (setup, preparação de dados, treinamento, build do Docker e execução local) através de comandos simplificados.

---

## 📁 Estrutura do Repositório

```text
Tech_Challenge5/
├── api/
│   └── app.py              # Servidor, endpoints da API, inferência e monitoramento (FastAPI)
├── data/                   # Diretório de dados processados
├── models/                 # Artefatos serializados (modelo .keras e scaler .pkl)
├── src/
│   ├── data_prep.py        # Pipeline de ingestão e transformação (ETL)
│   └── train.py            # Script de treinamento LSTM e rastreamento MLflow
├── .dockerignore           # Arquivos ignorados pelo Docker
├── .gitignore              # Arquivos ignorados pelo Git
├── Dockerfile              # Receita para empacotamento da aplicação
├── Makefile                # Automação de comandos do pipeline
└── requirements.txt        # Dependências do projeto fixadas por versão (incluindo Prometheus e psutil)
```


---

## 🛠️ Como Executar o Projeto

Você pode rodar o projeto de duas formas: nativamente via ambiente virtual ou de maneira isolada utilizando o Docker.

### Pré-requisitos
* Python 3.12+ instalado.
* Docker Desktop rodando em segundo plano (caso opte pelo contêiner).

### Opção A: Execução Nativa (Pipeline Completo)

1. **Crie e ative o ambiente virtual:**
   ```bash
   python -m venv venv
   # No Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # No Linux/Mac:
   source venv/bin/activate
   
1. Instale as dependências:

Bash
make setup
# Ou manualmente: pip install -r requirements.txt

2. Execute a preparação dos dados:

Bash
make data
# Ou: python src/data_prep.py

3. Treine o modelo (registra no MLflow e salva os artefatos):

Bash
make train
# Ou: python src/train.py

4. Inicie a API localmente:

Bash
make api
# Ou: uvicorn api.app:app --reload

Opção B: Execução via Docker (Padrão de Produção)
Para simular o deploy em um ambiente corporativo conteinerizado:

1. Construa a imagem Docker:

Bash
make docker-build
# Ou: docker build -t tech-challenge-5:latest .

2. Execute o contêiner:

Bash
make docker-run
# Ou: docker run -p 8000:8000 tech-challenge-5:latest

Com a aplicação rodando (seja via Uvicorn ou Docker), acesse a documentação interativa da API (Swagger) em:

👉 http://localhost:8000/docs (ou utilize o link público no Render).

🔌 Endpoints da API
1. Consumindo a API (POST /predict)
A API espera um payload JSON contendo um array unidimensional com exatamente 60 valores numéricos representando o histórico de preços de fechamento.

URL: /predict

Método: POST

Exemplo de Requisição (Payload):

{
  "historical_data": [
    35.1, 35.2, 35.5, 34.9, 35.0, 35.3, 35.6, 35.8, 36.0, 36.1,
    36.5, 36.2, 35.9, 36.0, 36.4, 36.7, 36.9, 37.0, 37.2, 37.5,
    37.1, 36.8, 36.9, 37.3, 37.6, 37.8, 38.0, 37.9, 37.5, 37.7,
    38.1, 38.4, 38.2, 38.0, 38.3, 38.5, 38.8, 39.0, 39.1, 38.9,
    38.6, 38.7, 39.2, 39.5, 39.3, 39.1, 39.4, 39.7, 39.9, 40.0,
    39.8, 39.5, 39.7, 40.1, 40.4, 40.6, 40.8, 40.5, 40.2, 40.5
  ]
}

Exemplo de Resposta:

{
  "predicted_price": 36.98,
  "currency": "BRL",
  "symbol": "PETR4.SA"
}
2. Verificação de Saúde (GET /health)

Retorna o status operacional da API, confirma se os artefatos foram carregados e reporta métricas de uso de CPU e memória RAM.

3. Métricas de Produção (GET /metrics)

Expõe as métricas da aplicação no formato padrão do Prometheus para observabilidade em produção.
