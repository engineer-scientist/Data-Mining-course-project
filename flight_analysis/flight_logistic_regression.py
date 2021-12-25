
import os
import random
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from os.path import join
from matplotlib import pyplot as plt
from flight_analysis import paths


''' MICHAEL STYRON '''
def run_algo():

    l = []
    m = []
    s = []
    al = []
    l_scores = []
    m_scores = []
    s_scores = []
    all_classes = []
    for _ in range(0, 10):
        l_score, m_score, s_score, all_class = log_reg()
        l.append(l_score)
        m.append(m_score)
        s.append(s_score)
        al.append(all_class)
    for x in range(0, 4):
        l_score = 0
        m_score = 0
        s_score = 0
        all_class = 0
        for i in range(0, 10):
            l_score = l_score + l[i][x]
            m_score = m_score + m[i][x]
            s_score = s_score + s[i][x]
        l_scores.append(l_score / 10)
        m_scores.append(m_score / 10)
        s_scores.append(s_score / 10)
        if x != 3:
            for j in range(0, 10):
                all_class = all_class + al[j][x]
            all_classes.append(all_class / 10)

    log_plot(l_scores, m_scores, s_scores, all_classes)

''' MICHAEL STYRON '''
def log_reg():
    # finding the .csv folder with data
    folder = os.getcwd()
    df = pd.read_csv(paths.copa_normalized_file)
    # finding the correct collumns to be measured
    destination = df['ROTATION_AVAILABLE_TM']
    taxiIn = df['TAXI_IN_MINUTES']
    taxiOut = df['TAXI_OUT_MINUTES']
    blockHr = df['SCH_BLOCK_HR']
    origin = df['ORIG_CD']
    depDelay = df['Departure_Delay']
    df = df.drop('Departure_Delay', axis='columns')

    # drop columns that copa airlines might not have before hand
    df = df.drop(columns='ARR_DELAY_MINUTES')
    df = df.drop(columns='DEP_DELAY_MINUTES')
    # df = df.drop(columns='ARR_DELAY_MINUTES_DV')
    df = df.drop(columns='SCH_ARR_DTML_PTY')
    df = df.drop(columns='SCH_ARR_DTMZ')
    df = df.drop(columns='SCH_DEP_DTML_PTY')
    df = df.drop(columns='SCH_DEP_DTMZ')
    df = df.drop(columns='d_Date')
    df = df.drop(columns='d_Time')
    df = df.drop(columns='a_Date')
    df = df.drop(columns='a_Time')

    df = df.loc[:, ~df.columns.str.startswith('COUNT')]
    df = df.loc[:, ~df.columns.str.startswith('sch_')]
    df = df.loc[:, ~df.columns.str.startswith('act_')]
    df = df.loc[:, ~df.columns.str.startswith('d_')]
    df = df.loc[:, ~df.columns.str.startswith('a_')]
    df = df.loc[:, ~df.columns.str.startswith('ETA')]
    df = df.loc[:, ~df.columns.str.startswith('ETD')]
    df = df.loc[:, ~df.columns.str.startswith('IN')]
    df = df.loc[:, ~df.columns.str.startswith('OFF')]
    df = df.loc[:, ~df.columns.str.startswith('ON')]
    df = df.loc[:, ~df.columns.str.startswith('OUT')]

    # getting the values of the columns and making them usable by the model
    d = np.array(destination.values)
    ti = np.array(taxiIn.values)
    to = np.array(taxiOut.values)
    yd = np.array(depDelay.values)
    yb = np.array(blockHr.values)
    og = np.array(origin.values)
    all = np.array(df.values)

    l_scores = []
    m_scores = []
    s_scores = []
    all_classes = []

    test = random.sample(range(5000, 7750), 500)
    t = []
    i = 1
    # creating a test set where the y values are balanced and alternating, not currently used
    for x in test:
        if i % 2 == 1:
            if yd[x] == 0 and i < 200:
                t.append(x)
                i = i + 1
        else:
            if yd[x] == 1 and i < 200:
                t.append(x)
                i = i + 1

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 500):
        x1.append(d[i])
        y1.append(yd[i])

    for j in test:
        x2.append(d[j])
        y2.append(yd[j])

    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    s_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 2000):
        x1.append(d[i])
        y1.append(yd[i])

    for j in test:
        x2.append(d[j])
        y2.append(yd[j])

    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    m_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 5000):
        x1.append(d[i])
        y1.append(yd[i])

    for j in test:
        x2.append(d[j])
        y2.append(yd[j])

    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    l_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 500):
        x1.append([d[i], ti[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    s_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 2000):
        x1.append([d[i], ti[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    m_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 5000):
        x1.append([d[i], ti[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    l_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 500):
        x1.append([d[i], ti[i], to[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    s_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 2000):
        x1.append([d[i], ti[i], to[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    m_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 5000):
        x1.append([d[i], ti[i], to[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    l_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 500):
        x1.append([d[i], ti[i], to[i], og[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j], og[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    s_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 2000):
        x1.append([d[i], ti[i], to[i], og[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j], og[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    m_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 5000):
        x1.append([d[i], ti[i], to[i], og[i]])
        y1.append(yd[i])

    for j in test:
        x2.append([d[j], ti[j], to[j], og[j]])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    l_scores.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 500):
        x1.append(all[i])
        y1.append(yd[i])

    for j in test:
        x2.append(all[j])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    all_classes.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 2000):
        x1.append(all[i])
        y1.append(yd[i])

    for j in test:
        x2.append(all[j])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    all_classes.append(model.score(x2, y2))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, 5000):
        x1.append(all[i])
        y1.append(yd[i])

    for j in test:
        x2.append(all[j])
        y2.append(yd[j])

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    overSample = RandomOverSampler(sampling_strategy='minority')
    x1_over, y1_over = overSample.fit_resample(x1, y1)

    overSample = RandomOverSampler(sampling_strategy='minority')
    # x2_over, y2_over = overSample.fit_resample(x2, y2)

    # fitting a logistical regression model on the data
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x1_over, y1_over)
    all_classes.append(model.score(x2, y2))

    return l_scores, m_scores, s_scores, all_classes

''' MICHAEL STYRON '''
def log_plot(l_scores, m_scores, s_scores, all_classes):

    path = join(paths.plot_dir, 'LOGISTIC REGRESSION RESULTS')

    rng = np.arange(1, 5, 1)
    fig, ax = plt.subplots()
    ax.plot(rng, l_scores, label='Large')
    ax.plot(rng, m_scores, label='Medium')
    ax.plot(rng, s_scores, label='Small')
    ax.set_xlabel('X attributes')
    ax.set_ylabel('Model score')
    ax.set_title('Logistic Regression on Copa Airlines data')
    ax.legend()

    fig.savefig(join(path, "LogRegPlot" + ".png"))
    # fig.savefig('LogRegPlot.png')

    plt.clf()
    x_axis = ['Small', 'Medium', 'Large']
    plt.bar(x_axis, all_classes)
    plt.title('Logistic Regression on all attributes')
    plt.xlabel('Size of Training Set')
    plt.ylabel('Score')
    plt.savefig(join(path, "LogRegGraph" + ".png"))
    # plt.savefig('LogRegGraph.png')



