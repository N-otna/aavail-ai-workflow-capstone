# Aavail AI Workflow – Capstone Project
AI in Production Coursera course

## Part 3: Model deployment
In this final part, we embed and deploy both our prediction models into a Flask app framework and build an API with **training**, **prediction** and **logging** endpoints. The app also has a dashboard for model performance and data drift monitoring. All the app components can be found in the [aavail-app](https://github.com/N-otna/aavail-ai-workflow-capstone/tree/master/Part%203/aavail-app) subfolder of Part 3.

To deploy the app in a test environment:
```
~$ python app.py
```
And go to http://0.0.0.0:8080/ to see the web app.

Using **Docker**, we can bundle our API, model and unit tests into a container:

```
~$ docker build -t aavail-app .
```

Run the container to ensure that the app is working:
```
~$ docker run -p 4000:8080 aavail-app
```
And go to http://0.0.0.0:4000/ to see the web app.

### Training, predicting and logging
We built an API endpoint for training, predicting and logging. 

For training and saving the models (both Random Forest and XGBoost), the **training endpoint** can also take a "test" argument to only train a couple of models and save time for testing purposes. We set up the training API endpoint in a way that the scale, load and drift in data can be anticipated and models can be retrained at regular intervals if necessary, using new fresh data as input. The requests can be made using cURL. For example, the following command trains models:

```
~$ query='{"mode":"test"}'
~$ curl -X POST -H "Content-Type: application/json" -d "$query" http://localhost:8080/train
```

Please note that the port needs to be adjusted (either `8080` or `4000` depending on whether you deployed with Docker or not).

The API **endpoint for predictions** takes in date, country name and model arguments to ouptut a monthly revenue prediction. Example of cURL command to obtain the 30 day revenue prediction based on the *5th of January 2018* for the *United Kingdom* using the *Random Forest* model (use `sl` for Random Forest and `xgb` for XGBoost predictions):

```
~$ query='{"query":{"country": "united_kingdom", "year": "2018", "month": "01", "day": "05", "model": "sl"}, "type":"dict"}'
~$ curl -X POST -H "Content-Type: application/json" -d "$query" http://localhost:8080/predict
```
This command outputs a predicted revenue in JSON: `{"y_pred":[157404.46399999995],"y_proba":"none"}`

For the **logging endpoint**, we can for example retrieve a particular log file using:
```
~$ curl -X GET -H "Content-Type: application/json" http://localhost:8080/logs/train-test.log
```
which will output the requested log in the form: 
```
unique_id,timestamp,tag,date,eval_test,model_version,model_version_note,runtime
2f4368ad-60a2-4293-b978-60dd4a37664d,1601375783.20294,united_kingdom,1600782826.2454798,{'rmse': 0.5},0.1,test model,00:00:01
```

### Unit tests
All main scripts, including the data ingestion and transformation (`cslib.py`) modelling (`model.py`) and logging (`logger.py`) contain basic test procedures embedded at the end of the script. If you run those separatly in Python it will automatically run a sanity test to ensure that the scripts' components are doing their tasks and working properly.

Additionally, we set up separate unit tests for the API, the models and the logging system.

The **API unit test** will test the essential functionalities of the `train`, `predict` and `log` endpoints. To run the API tests:
```
~$ python unittests/ApiTests.py
```

The **model unit test** will test that the data load, training and predicting functionalities all work as expected:
```
~$ python unittests/ModelTests.py
```

And the **logger unit test** will ensure that the log files both for training and for predictions are created and also ensure that those log files are properly stored and can be retrieved:
```
~$ python unittests/LoggerTests.py
```

To run **all three unit tests at once** please run the `run-tests.py` script:
```
~$ python run-tests.py
```

### Performance monitoring
We also set up a monitoring system to investigate the performance of our models and the quality of the incoming data, looking to determine and flag some potential drift. 

The `monitoring.py` script will load the data used in the latest training and determine outlier and distance thresholds. The outputs can be visualised in the following form:
![Outliers](https://github.com/N-otna/aavail-ai-workflow-capstone/blob/master/Part%203/monitoring_outliers.png)

