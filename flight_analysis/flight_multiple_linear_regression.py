from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flight_analysis import paths
import os
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt


''' AMANDA LOVETT '''
def run_algo():

    inputArray_Departure = ['AIRCRAFT_TYPE', 'CREW_CNT', 'FLT_ACTUAL_HR',
                            'TAXI_IN_MINUTES', 'TAXI_OUT_MINUTES', 'ORIG_CD', 'SCH_DEST_CD']

    inputArray_Arrival = ['AIRCRAFT_TYPE', 'DEP_DELAY_MINUTES', 'FLT_ACTUAL_HR', 'TAXI_IN_MINUTES', 'TAXI_OUT_MINUTES',
                          'ORIG_CD', 'SCH_DEST_CD']

    # resultArray_Departure = ['COUNT_DEP00', 'COUNT_DEP05', 'COUNT_DEP15', 'COUNT_DEP80',
    #                          'DEP_DELAY_MINUTES']  # result, what we're looking to obtain accurately
    #
    # resultArray_Arrival = ['COUNT_ARR00', 'COUNT_ARR14', 'COUNT_ARR15', 'COUNT_ARR90', 'ARR_DELAY_MINUTES']

    resultArray_Departure = ['DEP_DELAY_MINUTES']  # result, what we're looking to obtain accurately
    resultArray_Arrival = ['ARR_DELAY_MINUTES']

    train_model(inputArray_Arrival, resultArray_Arrival, "Arrival Delay Scores", "Arrival")
    train_model(inputArray_Departure, resultArray_Departure, "Departure Delay Scores", "Departure")


def train_model(inputArray, resultArray, filename, ylab):

    directory = os.getcwd()
    df = pd.read_csv(paths.copa_airlines_cleaned_file)

    X = df[inputArray]

    r2Array = []

    # for x in range(5):

    Y = df[resultArray[0]]

    scoreSum = 0
    # for y in range(500):
    x_train, x_test, y_train, y_test = train_test_split(X, Y)  # splits dataset into testing and training sets

    model = LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # each time running the model yields different result
    from sklearn.metrics import r2_score

    score = r2_score(y_test, y_pred)
    # scoreSum += score
    print(ylab + " Delay R2 score: " + str(score)) #;  # print average R2 score after 100 iterations
    # r2Array.append(scoreSum / 500)

    # fig, axs = plt.subplots(figsize=(10, 6))
    # axs.bar(resultArray, r2Array)
    # axs.set_ylabel("R2 Score")
    plt.plot([i for i in range(len(y_test))], y_test, label='observed')
    plt.plot([i for i in range(len(y_pred))], y_pred, label='predicted')

    plt.ylabel(ylab + ' Delay in Minutes')
    plt.xlabel('Flights')
    plt.title("Multiple Linear Regression Results")
    plt.legend()
    path = join(paths.plot_dir, 'LINEAR REGRESSION RESULTS')
    plt.savefig(join(path, filename + ".png"))
    plt.clf()
    # plt.show()



