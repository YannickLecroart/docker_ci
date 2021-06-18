FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy joblib==0.17.0

COPY train.py ./train.py

ENTRYPOINT [ "python", "train.py" ]