# Aavail AI Workflow – Capstone Project
AI in Production Coursera course

## Part 1: Data ingestion
In order to help stabilise staffing and budget projections, Aavail has asked us to build a service capable of projecting their revenues accross different countries using historical service purchases. To achieve this, we will use time series forecasting and fit different models on this historical data, using them to predict future observations.

Ideally, to achieve our objective, the transactional data should consist of features that contain information on the transaction date, location as well as a target variable that will reflect the generated revenue by each transaction (e.g. the price paid by the customer). If available, additional information on the customer and the service being purchased may be useful to refine our analysis and feed more information into our predictive models, in hopes of increasing the predictive accuracy and understanding of underlying patterns.

After ingesting the input data available in JSON files, the different columns needed to be uniformly formated. We then used a date range to ensure all days were accounted for in the data and extracted the top 10 countries with regards to the generated revenue. 

From this point we could start our **exploratory data analysis** (EDA). Among other things, we looked at the relationship between the target (i.e. the revenue) and the other features. The monthly revenue combined across all top 10 countries is depicted below as we were looking for potential trends or seasonal peaks in the data, where consumers would purchase or stream more. We noticed possible dips between summer and winter seasons while the revenue peaked in December 2019 – but the amount of data was insufficient to draw any significant conclusion on seasonal patterns.
![Monthly Revenues](https://github.ibm.com/Anton-Ryjov/aavail-ai-workflow-capstone/blob/master/Part%201/monthly_revenue_top10.png)

Additionally, we compared the **performance between the top 10 countries** with regards to the overall views, the revenue and the number of streams, with the United Kingdom standing out and accounting for the largest proportion of all three metrics (over 90%) while Hong Kong and the Netherlands were the least performing.

![Top 10 countries performances](https://github.ibm.com/Anton-Ryjov/aavail-ai-workflow-capstone/blob/master/Part%201/top_countries_metrics.png)

Additional graphs depicting the **relationship between the different variables** as well as the EDA script can be found in [Part 1](https://github.ibm.com/Anton-Ryjov/aavail-ai-workflow-capstone/tree/master/Part%201) of this repository. The data ingestion has been fully automated and exists as a script with functions in the `cslib.py` file.

