Use latest anaconda with python 3.12
Install dependencies
```
pip install pip install mlflow jupyter pandas numpy scikit-learn xgboost hyperopt typing pyarrow 
pip install zenml zenml[server]
```
Start mlflow and zenml services. Check the output for services URL
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models
zenml login --local
```
Run the pipeline. 
```
python duration-prediction.py
```
Stop the services.
```
zenml logout --local
```