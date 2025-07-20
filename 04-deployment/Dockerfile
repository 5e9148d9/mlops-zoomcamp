FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY starter.py Pipfile Pipfile.lock /app/

RUN pip install pipenv &&\
    pipenv install --deploy --system

WORKDIR /app

ENTRYPOINT ["python", "starter.py"]