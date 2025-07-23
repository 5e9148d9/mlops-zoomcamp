FROM python:3.10.9-slim

RUN pip install -U pip &&\
    pip install pipenv &&\
    pipenv install --system --deploy

COPY [ "model.bin", "batch.py", "Pipfile", "Pipfile.lock", "./" ]

ENTRYPOINT [ "python", "batch.py" ]