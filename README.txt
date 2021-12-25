Machine Learning for Flight Delay Forecasting
Sarthak Sharma, Abbey Centers, Amanda Lovett, and Michael Styron
CAP 5771 Data Mining Fall 2021
README File for flight_delay_forecasting.py
test
To run all analysis scripts that were created to analyze the dataset provided by Copa Airlines, please ensure the following:

1. flight_delay_forecasting.py is in the Data-Mining-Final-Project-Fall-2021 directory.

2. You are using Python 3.9 and the following python libraries are installed.
    scikit-learn
    numpy
    pandas
    matplotlib
    statistics
    datetime
    imblearn
    random
    keras
    torch

3. Start the analysis by running the flight_delay_forecasting.py python script. This program will import many different python files from our flight_analysis python package, which includes all scripts that were used to clean and derive variations of the original Copa Airlines dataset (FSU_Fully_Cleaned.csv), train many different machine learning algorithms on these datasets, evaluate the results, and save those results in the TESTRUN folder.

4. NOTE: Before running flight_delay_forecasting.py, the only dataset that should be in the Data-Mining-Final-Project-Fall-2021 directory should be the FSU_Fully_Cleaned.csv file. Additionally, the subfolders inside the TESTRUN directory should all be empty. By the end of execution, flight_delay_forecasting will have created three additional datasets based on FSU_Fully_Cleaned.csv and will have populated all subfolders in the TESTRUN directory with the results of our analysis.

5. The three additional datasets should be created in the Data-Mining-Final-Project-Fall-2021 directory and should be called Copa_Airlines_Cleaned.csv, Cleaned_Normalized.csv, and Copa_Airlines_per_day.csv.

    Copa_Airlines_Cleaned.csv is equivalent to FSU_Fully_Cleaned.csv but with the following modifications:
    -	Enumerated string values
    -	Removed columns with single value
    -	Removed some outlier rows with negative date time (only 3)
    -	Normalized values to lie within 0 and 1 to prevent bias
    -	Dropped rows where data is missing
    -	Splitted time columns for individual attributes

    Cleaned_Normalized.csv is equivalent to Copa_Airlines_Cleaned.csv but all attributes are normalized.
    Copa_Airlines_per_day.csv is a resampled version of Copa_Airlines_Cleaned.csv with various new attributes describing the status of the flight volume and flights delayed at PTY (Panama City) airport, which is the central hub airport at Copa airlines.

6. The EXPECTED folder contains the expected results from running flight_delay_forecasting.py, including the three derived datasets and the contents of the TESTRUN directory.

7. The time to execute is approximately 33 minutes. Print statements correspond to various steps of analysis.

8. By the end of execution, TESTRUN should include the following in each subfolder:

    ANALYSIS: 7 plots describing attribute correlations and distribution of the original dataset provided by Copa Airlines
    LINEAR REGRESSION RESULTS: 2 plots showing the results of multiple linear regression
    LOGISTIC REGRESSION RESULTS: 2 plots showing the results of logistic regression
    LSTM RESULTS TRAIN TEST SPLIT: 11 plots showing the result of training the long short-term memory neural network on the first 66% of the daily dataset and using the model to predict the remaining 33% for each attribute. A single attribute is dislayed in each plot and the training set and test set loss over each epoch of training is also included in its own plot
    LSTM WALK FORWARD VALIDATION: 10 plots showing the results of training the LSTM model on the first 33% and using walk forward validation to prediction the remaining 66%. Each attribute with its predictions are displayed separately in their own plot.
    NEURAL NETWORK RESULTS: 2 text files describing the status of 2 different artificial neural network models after each epoch of training
    REGRESSION MAE RESULTS: A bar plot displaying the mean absolute error results for multiple regression models at lookback windows of 1-20 when predicting the last 66% using walk-forward validation.