# Aavail AI Workflow – Capstone Project
AI in Production Coursera course

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
