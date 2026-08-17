.PHONY: setup data train api docker-build docker-run

setup:
	pip install -r requirements.txt

data:
	python src/data_prep.py

train:
	python src/train.py

api:
	uvicorn api.app:app --reload

docker-build:
	docker build -t tech-challenge-5:latest .

docker-run:
	docker run -p 8000:8000 tech-challenge-5:latest