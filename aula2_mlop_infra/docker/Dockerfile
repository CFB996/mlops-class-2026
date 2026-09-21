# Serves the bonsai Classifier over HTTP.
#
# Python 3.13 to match the JupyterLab image that trains the model — see Dockerfile.jupyter.
FROM python:3.13-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app.py ./api/app.py

EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
