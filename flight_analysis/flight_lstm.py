import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from flight_analysis import paths
from os.path import join
from matplotlib import pyplot as plt


''' ABBEY CENTERS - train lstm on 66% of "Per Day" dataset with 33% as validation set and plot predictions and train/test loss'''
def run_lstm():

    print("Training LSTM on first 66% with last 33% as validation set and computing loss over 40 epochs of training.")

    dataframe = read_csv(paths.copa_per_day_file, engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    dataset_cp = dataset.copy()

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    num_features = dataset.shape[1]
    num_epochs = 40
    look_back = 16
    num_forecasts = 1
    neurons = 8
    dropout = 0.2

    y_size = num_features * num_forecasts
    x_size = num_features * look_back
    data_supervised = series_to_supervised(dataset, n_in=look_back,
                                           n_out=num_forecasts)  # send link to how this function works!


    # split into train and test sets
    train_size = int(len(data_supervised) * 0.66)
    test_size = len(data_supervised) - train_size
    trainX, trainY, testX, testY = data_supervised[0:train_size, 0:-y_size], data_supervised[0:train_size, -y_size:], \
                                   data_supervised[train_size:len(data_supervised), 0:-y_size], data_supervised[
                                                                                                train_size:len(
                                                                                                    data_supervised),
                                                                                                -y_size:]
    batch_size = 1

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], num_forecasts, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], num_forecasts, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, num_forecasts, trainX.shape[2]), stateful=True))
    model.add(Dropout(dropout))
    model.add(Dense(y_size))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    history = list()
    training_loss = list()
    test_loss = list()
    for n in range(num_epochs):
        hist = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False,
                         validation_data=(testX, testY))
        # Get training and test loss histories
        training_loss += hist.history['loss']
        test_loss += hist.history['val_loss']
        model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    testPredict = model.predict(testX, batch_size=batch_size)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # calculate mean squared error
    train_mae = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.2f MAE' % (train_mae))
    test_mae = mean_absolute_error(testY, testPredict)
    print('Test Score: %.2f MAE' % (test_mae))

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    start = len(trainPredict) + look_back
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:start, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[start:, :] = testPredict

    # create a plot for each feature to visualize predictions vs observations
    i = 0

    path = join(paths.plot_dir, 'LSTM RESULTS TRAIN TEST SPLIT')
    for k in dataframe.keys():
        fig, axs = plt.subplots(figsize=(10, 6))
        axs.plot(dataset_cp[:, i], 'r-', label='Observed')
        axs.plot(trainPredictPlot[:, i], 'g-', label='Training Set Predicted')
        axs.plot(testPredictPlot[:, i], 'b-', label='Test Set Predicted')
        axs.set_title(k)
        axs.set_xlabel("day")
        fig.legend()
        fig.savefig(join(path, str(k) + ".png"))
        i += 1

    # Visualize loss history
    fig, axs = plt.subplots(figsize=(10, 6))
    axs.plot(epoch_count, training_loss, 'r--', label='Training Loss')
    axs.plot(epoch_count, test_loss, 'b-', label='Test Loss')
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    fig.legend()
    fig.savefig(join(path, "TRAIN_TEST_LOSS.png"))
    # plt.show()

''' ABBEY CENTERS - train lstm on 33% of "Per Day" dataset and perform walk-forward validation on remaining 33% and compute error'''
def run_lstm_walk_forward_validate():

    print("Performing walk-forward validation using LSTM.")

    dataframe = read_csv(paths.copa_per_day_file, engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    dataset_cp = dataset.copy()

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    num_features = dataset.shape[1]

    num_epochs = 40
    look_back = 16
    num_forecasts = 1
    neurons = 8
    dropout = 0.2
    train_percentage = 0.33
    y_size = num_features * num_forecasts
    x_size = num_features * look_back
    data_supervised = series_to_supervised(dataset, n_in=look_back,
                                           n_out=num_forecasts)  # send link to how this function works!

    # split into train and test sets
    train_size = int(len(data_supervised) * train_percentage)
    test_size = len(data_supervised) - train_size

    # one-step predictions
    testPredict = list()

    for i in range(test_size):

        trainX, trainY, testX, testY = data_supervised[0:train_size, 0:-y_size], data_supervised[0:train_size,
                                                                                 -y_size:], \
                                       data_supervised[train_size:len(data_supervised), 0:-y_size], data_supervised[
                                                                                                    train_size:len(
                                                                                                        data_supervised),
                                                                                                    -y_size:]
        train_size += 1
        batch_size = 1
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], num_forecasts, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], num_forecasts, testX.shape[1]))
        y_size = num_features * num_forecasts

        # Configure LSTM model layers where input has the shape [samples, time steps, features]
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, num_forecasts, trainX.shape[2]), stateful=True))
        model.add(Dropout(dropout))
        model.add(Dense(y_size))
        model.compile(loss='mean_absolute_error', optimizer='adam')

        history = list()
        training_loss = list()
        test_loss = list()
        for n in range(num_epochs):
            hist = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0,
                             shuffle=False)  # , validation_data=(testX, testY))
            # Get training and test loss histories
            training_loss += hist.history['loss']
            # test_loss += hist.history['val_loss']
            model.reset_states()

        # make single prediction
        input = testX[0]
        input = numpy.reshape(input, (input.shape[0], num_forecasts, input.shape[1]))
        single_pred = model.predict(input, batch_size=batch_size)
        testPredict.append(single_pred[0])

    train_size = int(len(data_supervised) * train_percentage)
    test_size = len(data_supervised) - train_size
    trainX, trainY, testX, testY = data_supervised[0:train_size, 0:-y_size], data_supervised[0:train_size, -y_size:], \
                                   data_supervised[train_size:len(data_supervised), 0:-y_size], data_supervised[
                                                                                                train_size:len(
                                                                                                    data_supervised),
                                                                                                -y_size:]

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], num_forecasts, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], num_forecasts, testX.shape[1]))

    trainPredict = model.predict(trainX, batch_size=batch_size)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # calculate mean squared error
    train_mae = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.2f MAE' % (train_mae))
    test_mae = mean_absolute_error(testY, testPredict)
    print('Test Score: %.2f MAE' % (test_mae))

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    start = len(trainPredict) + look_back
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:start, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[start:, :] = testPredict

    # create a plot for each feature to visualize predictions vs observations
    i = 0
    path = join(paths.plot_dir, 'LSTM WALK FORWARD VALIDATION')
    for k in dataframe.keys():
        fig, axs = plt.subplots(figsize=(10, 6))
        axs.plot(dataset_cp[:, i], 'r-', label='Observed')
        axs.plot(trainPredictPlot[:, i], 'g-', label='Training Set Predicted')
        axs.plot(testPredictPlot[:, i], 'b-', label='Test Set Predicted')
        axs.set_title(k)
        axs.set_xlabel("day")
        fig.legend()
        fig.savefig(join(path, str(k) + ".png"))
        i += 1

    # Visualize loss history
    fig, axs = plt.subplots(figsize=(10, 6))
    axs.plot(epoch_count, training_loss, 'r--', label='Training Loss')
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    fig.legend()
    # plt.show()

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
