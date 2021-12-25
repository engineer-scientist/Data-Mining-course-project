
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from flight_analysis import paths
from os.path import join
from matplotlib import pyplot as plt




''' ABBEY CENTERS - on the "Per Day" dataset, train k nearest neighbors, svm, decision tree, and extra tree for lookback windows of 1-20 and plot accuracy values on bar plot '''
def run_algos():

    dataframe = read_csv(paths.copa_per_day_file, engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    num_features = dataset.shape[1]
    num_forecasts = 1
    all_maes = list()
    min_lb = 1
    max_lb = 21

    # GRID SEARCH THE LOOKBACK WINDOW
    for look_back in range(min_lb, max_lb):

        y_size = num_features * num_forecasts
        x_size = num_features * look_back
        data_supervised = series_to_supervised(dataset, n_in=look_back,
                                               n_out=num_forecasts)  # send link to how this function works!

        # split into train and test sets
        train_size = int(len(data_supervised) * 0.33)
        test_size = len(data_supervised) - train_size
        trainX, trainY, testX, testY = data_supervised[0:train_size, 0:-y_size], data_supervised[0:train_size,
                                                                                 -y_size:], \
                                       data_supervised[train_size:len(data_supervised), 0:-y_size], data_supervised[
                                                                                                    train_size:len(
                                                                                                        data_supervised),
                                                                                                    -y_size:]

        models = list()
        # grid search new a knn model
        knn2 = KNeighborsRegressor()  # create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': np.arange(1, 25)}  # use gridsearch to test all values for n_neighbors
        grid = GridSearchCV(knn2, param_grid, cv=5)  # fit model to data
        grid.fit(trainX, trainY)
        models.append(("KNeighborsRegressor", grid.best_estimator_))

        # grid search svr model
        param_grid = {'estimator__C': [0.1, 1, 10, 100], 'estimator__gamma': [1, 0.1, 0.01, 0.001],
                      'estimator__kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(MultiOutputRegressor(SVR()), param_grid, refit=True, verbose=2)
        grid.fit(trainX, trainY)
        models.append(("SVR", grid.best_estimator_))

        # grid search a decision tree model
        param_grid = {"criterion": ["mse", "mae"],
                      "min_samples_split": [10, 20, 40],
                      "max_depth": [2, 6, 8],
                      "min_samples_leaf": [20, 40, 100],
                      "max_leaf_nodes": [5, 20, 100],
                      }
        grid = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
        grid.fit(trainX, trainY)
        models.append(("DecisionTreeRegressor", grid.best_estimator_))

        # grid search an extra tree model
        param_grid = {"criterion": ["mse", "mae"],
                      "min_samples_split": [10, 20, 40],
                      "max_depth": [2, 6, 8],
                      "min_samples_leaf": [20, 40, 100],
                      "max_leaf_nodes": [5, 20, 100],
                      }
        grid = GridSearchCV(ExtraTreeRegressor(), param_grid, cv=5)
        grid.fit(trainX, trainY)
        models.append(("ExtraTreeRegressor", grid.best_estimator_))


        model_maes = list()
        for name, m in models:
            mae, y, yhat = walk_forward_validation(trainX, trainY, testX, testY, scaler, m)
            print('MAE using ' + str(name) + ': %.3f' % mae)
            model_maes.append(float(mae))
        all_maes.append(model_maes.copy())

    labels = [str(i) for i in range(min_lb, max_lb)]
    all_maes = np.array(all_maes)
    knr_means = all_maes[:, 0]
    svr_means = all_maes[:, 1]
    dtr_means = all_maes[:, 2]
    etr_means = all_maes[:, 3]

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width - width + width / 2, knr_means, width, label='k nearest neighbors')
    rects2 = ax.bar(x - width + width / 2, dtr_means, width, label='decision tree regressor')
    rects3 = ax.bar(x + width / 2, etr_means, width, label='extra tree regressor')
    rects4 = ax.bar(x + width + width / 2, svr_means, width, label='support vector')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Lookback Window')
    ax.set_title('Algorithm Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.tight_layout
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, -0.05),
              fancybox=True, shadow=True, ncol=2)
    fig.tight_layout()

    ax.set_ylim([72.2, 72.9])

    path = join(paths.plot_dir, 'REGRESSION MAE RESULTS')
    fig.savefig(join(path, "regression_maes" + ".png"))
    # plt.show()


'''
ABBEY CENTERS
HELPER FUNCTIONS
SOURCES:
https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
'''

# convert a series to a supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

# fit an random forest model and make a one step prediction
def forecast(trainX, trainY, testX, model):
    # clone the model configuration
    local_model = clone(model)
    # fit the model
    local_model.fit(trainX, trainY)
    # make a one-step prediction
    yhat = local_model.predict([testX])
    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(trainX, trainY, testX, testY, scaler, model):
    predictions = list()
    historyx = [x for x in trainX]
    historyy = [y for y in trainY]

    # step over each time-step in the test set
    for i in range(len(testX)):
        yhat = forecast(historyx, historyy, testX[i], model)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        historyx.append(testX[i])
        historyy.append(testY[i])

    # invert predictions
    testPredict = scaler.inverse_transform(predictions)
    testY = scaler.inverse_transform(testY)

    # prediction error
    error = mean_absolute_error(testY, predictions)
    return error, testY, testPredict

''' END OF HELPER FUNCTIONS FOR REGRESSION ANALYSIS '''

