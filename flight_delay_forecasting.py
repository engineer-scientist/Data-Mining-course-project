import time
from flight_analysis import flight_lstm
from flight_analysis import flight_multiple_linear_regression
from flight_analysis import flight_regression
from flight_analysis import flight_logistic_regression
from flight_analysis import create_copa_airlines_cleaned
from flight_analysis import create_per_day_dataset
from flight_analysis import flight_analysis_and_neural_network



if __name__ == '__main__':
    start = time.time()
    # make the following changes to the original FSU_Fully_Cleaned.csv file provided by Copa Airlines
    ''' 
        The following changes are made to the FSU_Fully_Cleaned.csv file
        1.	Enumerate string values
        2.	Remove columns with single value
        3.	Removed some outlier rows with negative date time (only 3) 
        4.	Normalized values to lie within 0 and 1 to prevent bias
        5.	Drop rows where data is missing
        6.	Split time columns for individual attribute correlation evaluation
    '''

    create_copa_airlines_cleaned.create_dataset(normalize=False)    # clean the original set and save the new file
    create_copa_airlines_cleaned.create_dataset(normalize=True)     # save another version with noramlized attributes

    # resample copa_airlines_cleaned (not normalized) to get one sample per day with new attributes
    create_per_day_dataset.create_dataset()

    ''' ABBEY CENTERS '''
    flight_lstm.run_lstm() # run lstm neural network on per day dataset with train and validation set and save loss and prediction plots
    flight_lstm.run_lstm_walk_forward_validate() # run lstm on per day dataset and save walk forward validation prediction results
    flight_regression.run_algos() # run a suite of regression algorithms on the per day dataset and save bar plot results at multiple look back windows

    ''' MICHAEL STYRON '''
    flight_logistic_regression.run_algo()    # run logistic regression on cleaned_normalized dataset

    ''' AMANDA LOVETT '''
    flight_multiple_linear_regression.run_algo()    # run multiple linear regression on copa_airlines_cleaned

    ''' SARTHAK SHARMA'''
    flight_analysis_and_neural_network.neural_network()  # run standard neural network on original, fsu_fully_cleaned

    end = time.time()
    print("Total execution time: " + str(end-start))


