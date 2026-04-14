FROM python:3.10-slim

WORKDIR /app

ARG RUN_ID

RUN pip install --no-cache-dir mlflow tensorflow scikit-learn pandas

RUN echo "simulatin modle downlod for mlflo run id: ${RUN_ID}" > model_download_log.txt