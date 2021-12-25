
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from flight_analysis import paths
from os.path import join
from flight_analysis import paths
import torch
from torch import nn
# from torch.utils.data import DataLoader


''' SARTHAK SHARMA '''
def neural_network():

    # path to the dataset
    fname = paths.fsu_fully_cleaned


    # ----------------------- Working with input data -------------------------------------

    DF_2D = pd.read_csv(fname)
    DF_2D = DF_2D[(DF_2D['DEST_CD'] == 'PTY') | (DF_2D['ORIG_CD'] == 'PTY')]
    DF_2D = DF_2D[DF_2D['ORIG_CD'] != DF_2D['DEST_CD']]
    DF_2D = DF_2D[(-1000 < DF_2D['ARR_DELAY_MINUTES']) & (DF_2D['ARR_DELAY_MINUTES'] < 1000)]
    DF_2D = DF_2D[(-1000 < DF_2D['DEP_DELAY_MINUTES']) & (DF_2D['DEP_DELAY_MINUTES'] < 1000)]
    Copa_2D = DF_2D

    arrival_delay_category_1D = np.zeros(DF_2D.shape[0], dtype=int)
    departure_delay_category_1D = np.zeros(DF_2D.shape[0], dtype=int)

    interval_extremes_1D = [-10000, 0, 15, 30, 45, 60, 10000]

    for i in range(len(interval_extremes_1D) - 1):
        arrival_delay_category_1D[(interval_extremes_1D[i] < DF_2D['ARR_DELAY_MINUTES']) & (
                    DF_2D['ARR_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i
        departure_delay_category_1D[(interval_extremes_1D[i] < DF_2D['DEP_DELAY_MINUTES']) & (
                    DF_2D['DEP_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i

    arr_del_cat_counts_2D = np.zeros((DF_2D.shape[0], len(interval_extremes_1D) - 1))
    dep_del_cat_counts_2D = np.zeros((DF_2D.shape[0], len(interval_extremes_1D) - 1))

    for i in range(len(interval_extremes_1D) - 1):
        arr_del_cat_counts_2D[arrival_delay_category_1D == i, i] = 1
        dep_del_cat_counts_2D[departure_delay_category_1D == i, i] = 1

    legend_names_1D = []

    for i in range(len(interval_extremes_1D) - 1):
        legend_names_1D.append(str(interval_extremes_1D[i] + 1) + " to " + str(interval_extremes_1D[i + 1]))

    print('Delay categories =', legend_names_1D, '.')
    print("Number of flights in different arrival delay categories:")
    print(np.sum(arr_del_cat_counts_2D, axis=0))
    print("Fractions of flights in different arrival delay categories:")
    print(np.sum(arr_del_cat_counts_2D, axis=0) / np.sum(arr_del_cat_counts_2D))

    plt.bar(legend_names_1D, np.sum(arr_del_cat_counts_2D, axis=0))
    plt.xlabel('Arrival delay category')
    plt.ylabel('Number of flights')
    plt.grid()
    plt.xticks(rotation=90)


    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Bar chart - flight distribution in arrival delay categories'), bbox_inches='tight')
    # plt.savefig('Bar chart - flight distribution in arrival delay categories', bbox_inches='tight')
    plt.clf()

    print(np.sum(dep_del_cat_counts_2D, axis=0))
    print(np.sum(dep_del_cat_counts_2D, axis=0) / np.sum(dep_del_cat_counts_2D))

    plt.bar(legend_names_1D, np.sum(dep_del_cat_counts_2D, axis=0))
    plt.xlabel('Departure delay category')
    plt.ylabel('Number of flights')
    plt.grid()
    plt.xticks(rotation=90)

    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Bar chart - flight distribution in departure delay categories'), bbox_inches='tight')
    # plt.savefig('Bar chart - flight distribution in departure delay categories', bbox_inches='tight')
    plt.clf()

    plt.pie(np.sum(arr_del_cat_counts_2D, axis=0), autopct='%1.1f%%')  # , labels = legend_names_1D)
    plt.legend(legend_names_1D, bbox_to_anchor=(1.5, 0.5), loc='center right')
    plt.title('Distribution of flights in arrival delay categories', loc='left')
    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Pie chart - flight distribution in arrival delay categories'), bbox_inches='tight')
    # plt.savefig('Pie chart - flight distribution in arrival delay categories', bbox_inches='tight')
    plt.clf()

    plt.pie(np.sum(dep_del_cat_counts_2D, axis=0), autopct='%1.1f%%')
    plt.legend(legend_names_1D, bbox_to_anchor=(1.5, 0.5), loc='center right')
    plt.title('Distribution of flights in departure delay categories', loc='left')
    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Pie chart - flight distribution in departure delay categories'), bbox_inches='tight')
    # plt.savefig('Pie chart - flight distribution in departure delay categories', bbox_inches='tight')
    plt.clf()

    plt.plot(DF_2D['DEP_DELAY_MINUTES'], '.')
    plt.xlabel('Row index (flight)')
    plt.ylabel('Departure delay (minutes)')
    plt.grid()
    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Plot - departure delays'), bbox_inches='tight')
    # plt.savefig('Plot - departure delays', bbox_inches='tight')
    plt.clf()

    plt.plot(DF_2D['ARR_DELAY_MINUTES'], '.')
    plt.xlabel('Row index (flight)')
    plt.ylabel('Arrival delay (minutes)')
    plt.grid()
    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Plot - arrival delays'), bbox_inches='tight')
    # plt.savefig('Plot - arrival delays', bbox_inches='tight')
    plt.clf()

    plt.plot(DF_2D['DEP_DELAY_MINUTES'], DF_2D['ARR_DELAY_MINUTES'], '.')
    plt.xlabel('Departure delay (minutes)')
    plt.ylabel('Arrival delay (minutes)')
    plt.grid()
    plt.axis('scaled')
    path = join(paths.plot_dir, 'ANALYSIS')
    plt.savefig(join(path, 'Plot - departure delay vs arrival delay'), bbox_inches='tight')
    # plt.savefig('Plot - departure delay vs arrival delay', bbox_inches='tight')
    plt.clf()

    DEP_DELAY_MINUTES_1D = DF_2D['DEP_DELAY_MINUTES'].values
    ARR_DELAY_MINUTES_1D = DF_2D['ARR_DELAY_MINUTES'].values
    mean_dep_delay = np.mean(DEP_DELAY_MINUTES_1D)
    mean_arr_delay = np.mean(ARR_DELAY_MINUTES_1D)
    dep_diff_1D = mean_dep_delay - DEP_DELAY_MINUTES_1D
    arr_diff_1D = mean_arr_delay - ARR_DELAY_MINUTES_1D
    correlation_coefficient = np.sum(dep_diff_1D * arr_diff_1D) / (
                (np.sum(dep_diff_1D ** 2) * np.sum(arr_diff_1D ** 2)) ** 0.5)
    print('correlation_coefficient =', correlation_coefficient)

    print('max(DF_2D[\'DEP_DELAY_MINUTES\']) =', max(DF_2D['DEP_DELAY_MINUTES']))
    print('min(DF_2D[\'DEP_DELAY_MINUTES\']) =', min(DF_2D['DEP_DELAY_MINUTES']))
    print('max(DF_2D[\'ARR_DELAY_MINUTES\']) =', max(DF_2D['ARR_DELAY_MINUTES']))
    print('min(DF_2D[\'ARR_DELAY_MINUTES\']) =', min(DF_2D['ARR_DELAY_MINUTES']))

    attribute_names_1D = ['BANK_CD', 'DEST_CD', 'FLT_ACTUAL_HR', 'FLT_NUM', 'HUB_STN', 'IN_DTMZ', 'OFF_DTMZ', 'ON_DTMZ',
                          'ORIG_CD', 'OUT_DTMZ', 'ROTATION_AVAILABLE_TM', 'ROTATION_PLANNED_TM', 'ROTATION_REAL_TM',
                          'SCH_ARR_DTMZ', 'SCH_ARR_DTZ', 'SCH_ARR_TMZ', 'SCH_BLOCK_HR', 'SCH_DEP_DTMZ', 'SCH_DEP_DTZ',
                          'SCH_DEP_TMZ', 'TAIL', 'TAXI_IN_MINUTES', 'TAXI_OUT_MINUTES', 'AIRCRAFT_TYPE_ICAO', 'IN',
                          'ON', 'OUT', 'OFF']
    attributes_2D = DF_2D[attribute_names_1D]

    origins_1D = list(set(DF_2D['ORIG_CD']))
    print('origins_1D =', origins_1D)
    print('len(origins_1D) =', len(origins_1D), '. \n')
    for i in range(len(origins_1D)):
        DF_2D.loc[attributes_2D['ORIG_CD'] == origins_1D[i], 'ORIG_CD'] = i

    destinations_1D = list(set(DF_2D['DEST_CD']))
    print('destinations_1D =', destinations_1D)
    print('len(destinations_1D) =', len(destinations_1D), '. \n')
    for i in range(len(destinations_1D)):
        DF_2D.loc[attributes_2D['DEST_CD'] == destinations_1D[i], 'DEST_CD'] = i

    bank_codes_1D = list(set(DF_2D['BANK_CD']))
    print('bank_codes_1D =', bank_codes_1D)
    print('len(bank_codes_1D) =', len(bank_codes_1D), '. \n')
    for i in range(len(bank_codes_1D)):
        DF_2D.loc[attributes_2D['BANK_CD'] == bank_codes_1D[i], 'BANK_CD'] = i

    hub_stations_1D = list(set(DF_2D['HUB_STN']))
    print('hub_stations_1D =', hub_stations_1D)
    print('len(hub_stations_1D) =', len(hub_stations_1D), '. \n')
    for i in range(len(hub_stations_1D)):
        DF_2D.loc[attributes_2D['HUB_STN'] == hub_stations_1D[i], 'HUB_STN'] = i

    tails_1D = list(set(DF_2D['TAIL']))
    print('tails_1D =', tails_1D)
    print('len(tails_1D) =', len(tails_1D), '. \n')
    for i in range(len(tails_1D)):
        DF_2D.loc[attributes_2D['TAIL'] == tails_1D[i], 'TAIL'] = i

    aircraft_types_1D = list(set(DF_2D['AIRCRAFT_TYPE_ICAO']))
    print('aircraft_types_1D =', aircraft_types_1D)
    print('len(aircraft_types_1D) =', len(aircraft_types_1D), '. \n')
    for i in range(len(aircraft_types_1D)):
        DF_2D.loc[attributes_2D['AIRCRAFT_TYPE_ICAO'] == aircraft_types_1D[i], 'AIRCRAFT_TYPE_ICAO'] = i

    OD_1D = list(set(DF_2D['OD']))
    print('OD_1D =', OD_1D)
    print('len(OD_1D) =', len(OD_1D), '. \n')

    DF_2D['IN_DTMZ'] = DF_2D['IN_DTMZ'] / (10 ** 13)
    DF_2D['OFF_DTMZ'] = DF_2D['OFF_DTMZ'] / (10 ** 13)
    DF_2D['ON_DTMZ'] = DF_2D['ON_DTMZ'] / (10 ** 13)
    DF_2D['OUT_DTMZ'] = DF_2D['OUT_DTMZ'] / (10 ** 13)
    DF_2D['SCH_ARR_DTMZ'] = DF_2D['SCH_ARR_DTMZ'] / (10 ** 13)
    DF_2D['SCH_DEP_DTMZ'] = DF_2D['SCH_DEP_DTMZ'] / (10 ** 13)

    months_1D = pd.to_numeric(DF_2D['SCH_ARR_DTZ'].str.slice(start=5, stop=7))
    days_1D = pd.to_numeric(DF_2D['SCH_ARR_DTZ'].str.slice(start=8))
    sch_arr_mdz_1D = months_1D - 1 + (days_1D / 30)
    DF_2D['SCH_ARR_DTZ'] = sch_arr_mdz_1D

    months_1D = pd.to_numeric(DF_2D['SCH_DEP_DTZ'].str.slice(start=5, stop=7))
    days_1D = pd.to_numeric(DF_2D['SCH_DEP_DTZ'].str.slice(start=8))
    sch_dep_mdz_1D = months_1D - 1 + (days_1D / 30)
    DF_2D['SCH_DEP_DTZ'] = sch_dep_mdz_1D

    hours_1D = pd.to_numeric(DF_2D['SCH_ARR_TMZ'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['SCH_ARR_TMZ'].str.slice(start=3))
    sch_arr_tmz_1D = hours_1D + (minutes_1D / 60)
    DF_2D['SCH_ARR_TMZ'] = sch_arr_tmz_1D

    hours_1D = pd.to_numeric(DF_2D['SCH_DEP_TMZ'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['SCH_DEP_TMZ'].str.slice(start=3))
    sch_dep_tmz_1D = hours_1D + (minutes_1D / 60)
    DF_2D['SCH_DEP_TMZ'] = sch_dep_tmz_1D

    hours_1D = pd.to_numeric(DF_2D['IN'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['IN'].str.slice(start=3))
    in_1D = hours_1D + (minutes_1D / 60)
    DF_2D['IN'] = in_1D

    hours_1D = pd.to_numeric(DF_2D['ON'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['ON'].str.slice(start=3))
    on_1D = hours_1D + (minutes_1D / 60)
    DF_2D['ON'] = on_1D

    hours_1D = pd.to_numeric(DF_2D['OUT'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['OUT'].str.slice(start=3))
    out_1D = hours_1D + (minutes_1D / 60)
    DF_2D['OUT'] = out_1D

    hours_1D = pd.to_numeric(DF_2D['OFF'].str.slice(start=0, stop=2))
    minutes_1D = pd.to_numeric(DF_2D['OFF'].str.slice(start=3))
    off_1D = hours_1D + (minutes_1D / 60)
    DF_2D['OFF'] = off_1D

    test_2D = DF_2D.sample(int(0.3 * DF_2D.shape[0]))
    test_attributes_2D = test_2D[attribute_names_1D]

    test_arr_del_cat_1D = np.zeros(test_2D.shape[0], dtype=int)
    test_dep_del_cat_1D = np.zeros(test_2D.shape[0], dtype=int)

    for i in range(len(interval_extremes_1D) - 1):
        test_arr_del_cat_1D[(interval_extremes_1D[i] < test_2D['ARR_DELAY_MINUTES']) & (
                    test_2D['ARR_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i
        test_dep_del_cat_1D[(interval_extremes_1D[i] < test_2D['DEP_DELAY_MINUTES']) & (
                    test_2D['DEP_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i

    train_2D = DF_2D.drop(test_2D.index)
    train_attributes_2D = train_2D[attribute_names_1D]

    train_arr_del_cat_1D = np.zeros(train_2D.shape[0], dtype=int)
    train_dep_del_cat_1D = np.zeros(train_2D.shape[0], dtype=int)

    for i in range(len(interval_extremes_1D) - 1):
        train_arr_del_cat_1D[(interval_extremes_1D[i] < train_2D['ARR_DELAY_MINUTES']) & (
                    train_2D['ARR_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i
        train_dep_del_cat_1D[(interval_extremes_1D[i] < train_2D['DEP_DELAY_MINUTES']) & (
                    train_2D['DEP_DELAY_MINUTES'] <= interval_extremes_1D[i + 1])] = i

    TA_2D = np.array(test_attributes_2D.values, dtype=np.float32)
    test_attributes_2D = torch.tensor(TA_2D)

    test_arr_del_cat_1D = torch.tensor(test_arr_del_cat_1D)
    test_dep_del_cat_1D = torch.tensor(test_dep_del_cat_1D)
    train_arr_del_cat_1D = torch.tensor(train_arr_del_cat_1D)
    train_dep_del_cat_1D = torch.tensor(train_dep_del_cat_1D)

    TA_2D = np.array(train_attributes_2D.values, dtype=np.float32)
    train_attributes_2D = torch.tensor(TA_2D)

    # --------------------- Creating ANN model -----------------------------

    # Get CPU or GPU device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    n_neurons_1 = 100
    n_neurons_2 = 100
    n_neurons_3 = 100

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(test_attributes_2D.shape[1], n_neurons_1),
                nn.ReLU(),
                nn.Linear(n_neurons_1, n_neurons_2),
                nn.ReLU(),
                nn.Linear(n_neurons_2, n_neurons_3),
                nn.ReLU(),
                nn.Linear(n_neurons_3, len(interval_extremes_1D) - 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print("ANN model = \n", model)

    # ----------------------- Optimizing model parameters --------------------

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(train_attributes_2D, train_labels_1D, model, loss_fn, optimizer, batch_size):
        size = len(train_labels_1D)
        num_batches = int(len(train_labels_1D) / batch_size) + 1
        model.train()
        for b in range(num_batches):
            # if b < num_batches - 1:
            X = train_attributes_2D[b * batch_size: (b + 1) * batch_size, :]
            y = train_labels_1D[b * batch_size: (b + 1) * batch_size]
            if b == num_batches - 1:
                X = train_attributes_2D[b * batch_size:, :]
                y = train_labels_1D[b * batch_size:]
            X = X.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            # print(type(pred))
            # print(type(y))
            y = y.type(torch.LongTensor)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 100 == 0:
                loss, current = loss.item(), b * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(test_attributes_2D, test_labels_1D, model, loss_fn, batch_size):
        size = len(test_labels_1D)
        num_batches = int(len(test_labels_1D) / batch_size) + 1
        model.eval()
        sum_test_loss = 0
        sum_correct = 0
        y_predicted = []
        with torch.no_grad():
            for b in range(num_batches):
                # if b < num_batches - 1:
                X = test_attributes_2D[b * batch_size: (b + 1) * batch_size, :]
                y = test_labels_1D[b * batch_size: (b + 1) * batch_size]
                if b == num_batches - 1:
                    X = test_attributes_2D[b * batch_size:, :]
                    y = test_labels_1D[b * batch_size:]
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                sum_test_loss = sum_test_loss + loss_fn(pred, y.type(torch.LongTensor)).item()
                sum_correct = sum_correct + (pred.argmax(1) == y).type(torch.float).sum().item()
                y_predicted.append(pred.argmax(1))
        avg_test_loss = sum_test_loss / num_batches
        avg_correct = sum_correct / size
        return 100 * avg_correct, avg_test_loss, y_predicted

    epochs = 10
    dep_avg_correct_1D = []
    dep_avg_test_loss_1D = []
    batch_size = 100
    predicted_dep_del_cat_1D = []


    path = join(paths.plot_dir, 'NEURAL NETWORK RESULTS')
    file_path = join(path, 'ANN_output_train_test_departure_delays.txt')
    with open(file_path, 'w') as out_file:

        for t in range(epochs):
            print(f"\n Epoch {t + 1}\n-------------------------------")
            out_file.write(f"\n Epoch {t + 1}\n")
            train(train_attributes_2D, train_dep_del_cat_1D, model, loss_fn, optimizer, batch_size)
            dep_avg_correct, dep_avg_test_loss, predicted_dep_del_cat_1D = test(test_attributes_2D, test_dep_del_cat_1D,
                                                                                model, loss_fn, batch_size)
            dep_avg_correct_1D.append(dep_avg_correct)
            dep_avg_test_loss_1D.append(dep_avg_test_loss)
            # predicted_dep_del_cat_1D.append(delay_cat)
            print(f"Test Error: \n Accuracy: {dep_avg_correct:>0.1f}%, Avg loss: {dep_avg_test_loss:>8f} \n")
            out_file.write(f"Test Error: \n Accuracy: {dep_avg_correct:>0.1f}%, Avg loss: {dep_avg_test_loss:>8f} \n")

        print("Done!")
        print("\n ---------------------------------------------------- \n")
        out_file.write("\n Done!")
        out_file.write("\n ---------------------------------------------------- \n")


    # !cat ANN_output_train_test_departure_delays.txt

    epochs = 10
    arr_avg_correct_1D = []
    arr_avg_test_loss_1D = []
    batch_size = 100
    predicted_arr_del_cat_1D = []

    path = join(paths.plot_dir, 'NEURAL NETWORK RESULTS')
    file_path = join(path, 'ANN_output_train_test_arrival_delays.txt')
    with open(file_path, 'w') as out_file:

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            out_file.write(f"\n Epoch {t + 1}\n")
            train(train_attributes_2D, train_arr_del_cat_1D, model, loss_fn, optimizer, batch_size)
            arr_avg_correct, arr_avg_test_loss, predicted_arr_del_cat_1D = test(test_attributes_2D, test_arr_del_cat_1D,
                                                                                model, loss_fn, batch_size)
            arr_avg_correct_1D.append(arr_avg_correct)
            arr_avg_test_loss_1D.append(arr_avg_test_loss)
            # predicted_dep_del_cat_1D.append(delay_cat)
            print(f"Test Error: \n Accuracy: {arr_avg_correct:>0.1f}%, Avg loss: {arr_avg_test_loss:>8f} \n")
            out_file.write(f"Test Error: \n Accuracy: {arr_avg_correct:>0.1f}%, Avg loss: {arr_avg_test_loss:>8f} \n")

        print("Done!")
        print("\n ---------------------------------------------------- \n")
        out_file.write("\n Done!")
        out_file.write("\n ---------------------------------------------------- \n")



    # !cat ANN_output_train_test_arrival_delays.txt


    # use below to save a figure to the correct folder
    # path = join(paths.plot_dir, 'NEURAL NETWORK RESULTS')
    # fig.savefig(join(path, "filename" + ".png"))






