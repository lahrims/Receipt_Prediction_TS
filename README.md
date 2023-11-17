I have trained two models, XGBoost and FBProphet (which I wanted to try for a long time). To explore these models, please refer to the notebooks file.

The journey began with the data analysis of this Time Series model, revealing that the data is non-stationary and exhibits seasonality. Consequently, I opted to implement XGBoost and FBProphet.

I saved the trained models using the pickle module and created an abstract class to facilitate the addition of more models to the code.

Initially, I attempted to build an LSTM, but the results were unsatisfactory. While it performed well for short-term predictions, I found myself recursively adding all the predictions. As a result, I decided to proceed with XGBoost.

I utilized autocorrelation to determine the number of lags required.

To run the model:

1. Clone the repository.

2. Build the Docker image by running:

   ```
   docker build -t receipt_pred 
   ```

   This command installs all the requirements for this project.

3. To witness the model in action, run:

   ```
   docker run -p 8080:5000 receipt_pred
   ```

These instructions will set up and run the Docker container, allowing you to observe the functionality of the model.
