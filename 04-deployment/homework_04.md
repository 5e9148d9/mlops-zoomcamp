In order to create pipenv it is better to run docker image and generate environment inside.
```
docker run -v "/tmp/04:/app2" -it --rm --entrypoint=bash agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
cd /app2
pipenv install scikit-learn==1.5.0 pandas pyarrow
```
Now Pipfile and Pipfile.lock are created, and we can deploy and test the `starter.py`.
```
pipenv install --system --deploy
cp /app/model.bin .
python starter.py --year=2023 --month=5
```
Copy Pipfile and Pipfile.lock from `/tmp/04`.