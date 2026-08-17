# 1. Usa uma imagem oficial do Python bem leve
FROM python:3.12-slim

# 2. Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# 3. Instala dependências do sistema necessárias para algumas bibliotecas de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia o arquivo de requisitos PRIMEIRO (isso deixa o build mais rápido no futuro)
COPY requirements.txt .

# 5. Instala as bibliotecas do Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia todo o resto do seu projeto para dentro do contêiner
COPY . .

# 7. Expõe a porta que a API vai usar
EXPOSE 8000

# 8. O comando exato que liga o servidor quando o contêiner iniciar
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]