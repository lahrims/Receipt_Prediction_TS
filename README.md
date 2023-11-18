# Introduction:

This project focuses on analyzing a time series dataset that captures the daily receipt counts throughout the year 2021. The primary objective is to train specific models using the 366 days' worth of receipt count data, assess their performance, and subsequently deploy them for accurate predictions of receipt counts in the year 2022.
Structure:
```
.
├── Dockerfile
├── README.md 
├── api
│   ├── app.py
│   ├── cfg
│   │   ├── ModelsCfg.py
│   │   └── __init__.py
│   ├── functions.py
│   ├── templates
│   │   ├── index.html
│   │   └── predict.html
│   └── utils.py
├── data
│   ├── data_daily.csv
│   └── df_xgb.csv
├── models
│   ├── __init__.py
│   ├── fbprophet.pckl
│   └── xgb.pckl
├── notebooks
│   └── Models_Training.ipynb
└── requirements.txt
```
# Prerequisites:
1.	The system must have Python version 3.11.4 or higher installed.
2.	Docker Desktop is required.
3.	Git installation is necessary.


# Installation:
## With Docker:
1.	Clone the Repo on your git bash
```
git clone https://github.com/lahrims/Receipt_Prediction_TS.git
```
2.	Start the Docker Engine
3.	Cd into the Receipt_Prediction_TS
```
cd ../path/
```
4.	Run the Docker build 
```
docker build -t receipt_pred
```
5.	This goes through requirements.txt and installs all the required libraries for this project 
6.	Now to Run the web application that rus with a simple flask and html rendering which predicts the receipt counts for the day
```
docker run -p 8080:5000 receipt_pred
```
7.	Open http://localhost:5000
8.	You Should be able to see this page
 <img width="1314" alt="Screen Shot 2023-11-17 at 8 57 44 AM" src="https://github.com/lahrims/Receipt_Prediction_TS/assets/68388027/6a0012ab-3077-4ed0-a568-9e24bd764617">
eipt counts. Use

10.	Select the model and Insert the date to predict 
```
Sample Input:
Model: XGBoost
Date:2022-01-01
```

10.	This should get the prediction
 
<img width="1291" alt="Screen Shot 2023-11-17 at 9 30 32 AM" src="https://github.com/lahrims/Receipt_Prediction_TS/assets/68388027/64e9de65-52aa-4beb-a968-c1958cbfe3b2">


## Without Docker:
1.	Follow steps until step 2
2.	Make sure to install all the required libraries from requirements.txt
3.	Now change directory to Receipt_Prediction_TS/api/
```
Flask run
```
4.	Flask automatically detects the flask file as it has 
```
flask.Flask(__name__)
```
5.	You should be able to see something like this,
```
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```
6.	Now Repeat the same steps from 8 to 10 after opening the localhost.


# Project Details:
## Data Analysis

1. Effective data analysis and preprocessing significantly impact model performance.Initiated the process by conducting data analysis on the dataset and visualizing the mean to assess the linearity in the time series
   ![plot_1](https://github.com/lahrims/Receipt_Prediction_TS/assets/68388027/1407999c-0fde-414d-a4a5-8aefc2a34625)
   The results distinctly reveal a consistent linear increase in the mean throughout the year 2021.
2. To develop a time series model, it is crucial to determine whether the data is stationary or non-stationary
      Plotted Rolling mean and Std
   ![AIC](https://github.com/lahrims/Receipt_Prediction_TS/assets/68388027/4f1fd62e-f86d-4a72-abc1-b26d068444c2)

Performed the Adfuller test to confirm that the data is non-sationary
```
    		Values                       Metric
	0    0.175044              Test Statistics
	1    0.970827                      p-value
	2   17.000000             No. of lags used
	3  347.000000  Number of observations used
	4   -3.449337          critical value (1%)
	5   -2.869906          critical value (5%)
	6   -2.571227         critical value (10%)
```
The probability value is clearly greater (>0.5) and proves that the dats is non-stationary
3.	Now to see if the past values of Receipt are affecting the present, defined a autocorreclation function which checks for autocorrelation between the lag features and it was clearly seen that the features are heavily auto correlated
```
	autocorrelation_lags 1 : 0.9247303360311816
	autocorrelation_lags 21 : 0.9112605951402099
	autocorrelation_lags 41 : 0.9110364794285617
	autocorrelation_lags 61 : 0.8890441999129224
	autocorrelation_lags 81 : 0.8623421804206457
	autocorrelation_lags 101 : 0.8486912819535416
	autocorrelation_lags 121 : 0.8239378317112811
	autocorrelation_lags 141 : 0.8198227803688078
	autocorrelation_lags 161 : 0.7832810815616251

```
4.	It is essential to check for seasonality and trend in the time series data set to select the model.
   ![Sesonality](https://github.com/lahrims/Receipt_Prediction_TS/assets/68388027/e7b83a4c-9f2d-4779-b73d-4d993dd2cda2)
The Data clearly Exhibits Sesonality and Trend!

Checking for seasonality and trend in a time series dataset is essential when selecting a model because these components can significantly influence the behavior of the data over time. Understanding the presence of seasonality and trend helps in choosing an appropriate model that can capture and account for these patterns.

## Model Selection

1. Initially, I trained the model using LSTM and Linear Regression with the Adam Optimizer. Although the model exhibited strong performance during training, the predictions posed challenges, as it could only foresee short-term outcomes. To address this limitation, it became necessary to iteratively incorporate the predictions back into the data.

2. Seeking alternatives for non-stationary, seasonal data characterized by a linearly increasing trend, I found that XGBoost and FBProphet were well-suited for such data patterns, enabling predictions over more extended periods.

3. The data was preprocessed separately for each of the models.

4. Essential lags and additional features were introduced using user-defined functions, and feature importance analysis was conducted to select the most influential features.


5. Conducted hyperparameter tuning on the models to optimize their performance and achieve improved results.

6. Following visualization and evaluation of performance metrics, it became evident that XGBoost outperformed FBProphet. XGBoost demonstrated adept learning of the data's seasonality with some fine-tuning in the learning rate. To enhance results further, k-fold validation was performed.

7. The Mean Squared Error (MSE) score for XGBoost decreased as the number of epochs increased and reached a plateau after 100 epochs. Upon testing on a future dataset, the model exhibited consistent seasonality and trend patterns similar to those observed in the provided 2021 data. This suggests that the XGBoost model effectively learned from the training data and was able to generalize well to new, unseen data, providing reliable predictions.

8. Saved the models using Pickle for future use.

## Inference

1. Implemented an abstract class in util.py, providing a flexible foundation for easy extension of the project by adding more models.

2. Developed Data Preprocessing, Post Processing, and Output functions in a generalized manner, allowing for easy modification specific to each model. These functions are designed to accept the target date as input.

3. Ensured that data is appropriately formatted as inputs for both XGBoost and FBProphet through the use of Predictor classes.

4. Conducted testing with sample inputs as mentioned earlier to validate the model's functionality.

5. Created Useful User Defined functions for preprocessing steps at functions.py

6. Upon successful retrieval of predictions by the model, proceeded to develop the Flask interface for improved user interaction and accessibility.

## Flask

1. Loaded saved models from ModelsCfg.py, employing a straightforward for loop to select models specified in index.html.
   
2. I choose flask as it is a popular and lightweight web framework for Python, and its simplicity makes it a good choice for developing web applications, especially for projects with specific requirements or when simplicity and flexibility are key considerations.

3. Initialized Flask in app.py, established app routes for both the index page and predictions page, and incorporated logic in util.py to navigate through the predictor classes.

4. Implemented basic CSS styling for both the Index and Predictor pages to enhance the visual presentation.

5. Dockerized the project for easier deployement and Isolation of required Dependencies.

# Future Scope:

1. The project holds potential for extension by incorporating various other models to diversify the modeling approaches and explore their performance on the given dataset.

2. Implementing validation sets can contribute to the improvement of models by allowing for better assessment and fine-tuning during the training process.

3. There is an opportunity to expand the project by integrating Long Short-Term Memory (LSTM) or Multi-Layer Perceptron (MLP) models to assess their performance, especially considering the limited one-year dataset. This extension could provide insights into the comparative effectiveness of different model architectures.




   

