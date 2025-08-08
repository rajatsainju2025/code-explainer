FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .[web]

EXPOSE 7860
CMD ["python", "-m", "code_explainer.cli", "serve", "--host", "0.0.0.0", "--port", "7860"]
