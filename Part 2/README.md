# Aavail AI Workflow – Capstone Project
AI in Production Coursera course

## Part 1: Data ingestion
In order to help stabilise staffing and budget projections, Aavail has asked us to build a service capable of projecting their revenues accross different countries using historical service purchases. To achieve this, we will use time series forecasting and fit different models on this historical data, using them to predict future observations.

Ideally, to achieve our objective, the transactional data should consist of features that contain information on the transaction date, location as well as a target variable that will reflect the generated revenue by each transaction (e.g. the price paid by the customer). If available, additional information on the customer and the service being purchased may be useful to refine our analysis and feed more information into our predictive models, in hopes of increasing the predictive accuracy and understanding of underlying patterns.

After ingesting the input data available in JSON files, the different columns needed to be uniformly formated. We then used a date range to ensure all days were accounted for in the data and extracted the top 10 countries with regards to the generated revenue. 

From this point we could start our **exploratory data analysis** (EDA). Among other things, we looked at the relationship between the target (i.e. the revenue) and the other features. The monthly revenue combined across all top 10 countries is depicted below as we were looking for potential trends or seasonal peaks in the data, where consumers would purchase or stream more. We noticed possible dips between summer and winter seasons while the revenue peaked in December 2019 – but the amount of data was insufficient to draw any significant conclusion on seasonal patterns.
![Monthly Revenues](https://github.com/N-otna/aavail-ai-workflow-capstone/blob/master/Part%201/monthly_revenue_top10.png)

Additionally, we compared the **performance between the top 10 countries** with regards to the overall views, the revenue and the number of streams, with the United Kingdom standing out and accounting for the largest proportion of all three metrics (over 90%) while Hong Kong and the Netherlands were the least performing.

![Top 10 countries performances](https://github.com/N-otna/aavail-ai-workflow-capstone/blob/master/Part%201/top_countries_metrics.png)

Additional graphs depicting the **relationship between the different variables** as well as the EDA script can be found in [Part 1](https://github.com/N-otna/aavail-ai-workflow-capstone/tree/master/Part%201) of this repository. The data ingestion has been fully automated and exists as a script with functions in the `cslib.py` file.

## Part 2: Modelling
For the time series forecasting, we implemented two prediction that combine the output of multiple decision trees: a **Random Forest** and an **XGBoost regressor**. 

To provide inputs that our machine learning models could use to capture the underlying patterns or trends in the the data, we **engineered the features** so that for any given day the target becomes the sum of the next days revenue. We aggregate the transactions by day and use windows in time back from a specific date. Finally, we also add some features not directly related to the revenue, such as the 'recent views' and the 'recent invoices'. The extent of our feature engineering can be seen in the `cslib.py` script in the [Part 2](https://github.com/N-otna/aavail-ai-workflow-capstone/tree/master/Part%202) folder. We end up with a total of **7 explanatory variables** (i.e. features) to feed our models:
> previous_7, previous_14, previous_28, previous_70, previous_year, recent_invoices and recent_views

while the **30-day revenue** remains our target variable.

In the modelling phase, for both the Random Forest and the XGBoost, we used a pipeline architecture with a standard scaler transformation for both, such that our input data distribution will have a mean value 0 and standard deviation of 1. We split the data into train and test subsets first. Then, for **hyperparameter tuning**, we used a [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) with 5-fold cross validation to exhaustively consider all parameter combinations. For the XGBoost, on the other hand, we implemented an alternative way of tuning hyperparameters, namely a [randomised search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). In contrast to the grid search, this method tries out a fixed number of parameter settings sampled form specified distributions which greatly increases the efficiency and reduces our training time.

After engineering the features and shaping the pipeline architecture as well as the parameter tuning, we retrain both models on all of the available data and compared their performances. Looking at the prediction errors, the Random Forest performed better with a narrower distribution around 0.
![Prediction errors](https://github.com/N-otna/aavail-ai-workflow-capstone/blob/master/Part%202/error_hist.png)

In terms of Root Mean Square Error (RMSE), we can also see that the Random Forest performed better as it generated a lower RMSE on its predictions.
![RMSE](https://github.com/N-otna/aavail-ai-workflow-capstone/blob/master/Part%202/rmse.png)

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

