# explore the effect of the variance thresholds on the number of selected features
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean
import datetime
from os.path import join
from flight_analysis import paths

pattern = '%Y-%m-%d'

''' 
Abbey Centers
Resample Copa_Airlines_Cleaned to create the Copa_Airlines_per_day.csv file.
BE SURE TO RUN create_copa_airlines_cleaned.py BEFORE RUNNING THIS SCRIPT
'''


'''
total flight volume, departure flight volume, arrival flight volume, the 
number of flight delay, delay rate, departure delay rate , 
arrival delay rate, average delay time, average departure 
delay time, average arrival delay time. 
'''


def create_dataset():
    ''' ASSIGN FNAME WITH PATH TO COPA_AIRLINES_CLEANED.CSV'''
    fname = paths.copa_airlines_cleaned_file
    df = pd.read_csv(fname)

    plot = False
    dr = delay_rate_per_day(df, "flight delay rate per day at PTY", plot)
    adt = avg_delay_time_per_day(df, "average delay time per day at PTY", plot)
    v = volume_per_day(df, "total arrivals and departures per day at PTY", plot)
    fd = flights_delayed_per_day(df, "total flights delayed per day at PTY", plot)
    a_dr = arr_delay_rate_per_day(df, "arrival delay rate per day at PTY", plot)
    a_adt = avg_arr_delay_time_per_day(df, "average arrival delay time per day at PTY", plot)
    a_v = arr_volume_per_day(df, "total arrivals per day at PTY", plot)
    d_dr = dep_delay_rate_per_day(df, "departure delay rate per day at PTY", plot)
    d_adt = avg_dep_delay_time_per_day(df, "average departure delay time per day at PTY", plot)
    d_v = dep_volume_per_day(df, "total departures per day at PTY", plot)
    mddtpd = max_dep_delay_time_per_day(df, "max departure delay time per day", plot)
    madtpd = max_arr_delay_time_per_day(df, "max departure delay time per day", plot)

    # new dataframe with added features
    new_df = pd.DataFrame()
    new_df['flights_delayed_per_day'] = fd
    new_df['arr_delay_rate_per_day'] = a_dr
    new_df['avg_arr_delay_time_per_day'] = a_adt
    new_df['delay_rate_per_day'] = dr
    new_df['avg_delay_time_per_day'] = adt
    new_df['dep_delay_rate_per_day'] = d_dr
    new_df['avg_dep_delay_time_per_day'] = d_adt
    new_df['volume_per_day'] = v
    new_df['arr_volume_per_day'] = a_v
    new_df['dep_volume_per_day'] = d_v

    new_df.to_csv(join(paths.working_dir, r"Copa_Airlines_per_day.csv."), index=False)
    plt.clf()


# total number of flights that depart or arrival at PTY per day
def volume_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # get load per day
        load_per_day.append(len(dfo_u) + len(dfd_u))

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights with departure or arrival delay > 0 for a single day / total flights that departed or arrived at PTY for that day
def delay_rate_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for records where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # flights delayed on departure where PTY is origin airport
        h_dfo_d = dfo_u[dfo_u['DEP_DELAY_MINUTES'] > 0]

        # flights delayed on arrival where PTY is destination airport
        h_dfd_d = dfd_u[dfd_u['ARR_DELAY_MINUTES'] > 0]

        # get delay rate i.e. (# flights delayed on arrival at PTY + # flights delayed on departuren from PTY) /
        total_delayed = len(h_dfo_d) + len(h_dfd_d)
        total = len(dfo_u) + len(dfd_u)
        if total > 0:
            delay_rate = total_delayed / total
        else:
            delay_rate = 0
        load_per_day.append(delay_rate)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# sum of all arrival and departure delay times for all flights that arrived or departed from PTY per day / number of flights that arrived or departed from PTY for that hour
def avg_delay_time_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # flights delayed on departure where PTY is origin airport
        delay_time = list()
        for index, row in dfo_u.iterrows():
            delay_time.append(row['DEP_DELAY_MINUTES'])

        # flights delayed on arrival where PTY is destination airport
        for index, row in dfd_u.iterrows():
            delay_time.append(row['ARR_DELAY_MINUTES'])

        if len(delay_time) > 0:
            avg_delay = mean(delay_time)
        else:
            avg_delay = 0

        load_per_day.append(avg_delay)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights that departed or arrived at PTY with dep or arr delay > 0 per day
def flights_delayed_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # flights delayed on departure where PTY is origin airport
        h_dfo_d = dfo_u[dfo_u['DEP_DELAY_MINUTES'] > 0]

        # flights delayed on arrival where PTY is destination airport
        h_dfd_d = dfd_u[dfd_u['ARR_DELAY_MINUTES'] > 0]

        load_per_day.append(len(h_dfo_d) + len(h_dfd_d))

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights that departed from PTY per day
def dep_volume_per_day(df_a, label, plot):
    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # get load per day
        load_per_day.append(len(dfo_u))

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights that arrived at PTY per day
def arr_volume_per_day(df_a, label, plot):
    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # get load per day
        load_per_day.append(len(dfd_u))

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights that departed from PTY with dep delay > 0 / total number of flights that departed from PTY per day
def dep_delay_rate_per_day(df_a, label, plot):
    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # flights delayed on departure where PTY is origin airport
        h_dfo_d = dfo_u[dfo_u['DEP_DELAY_MINUTES'] > 0]

        # get delay rate i.e. (# flights delayed on arrival at PTY + # flights delayed on departuren from PTY) /
        # (total flights arriving at PTY + total flights leaving PTY)
        total_delayed = len(h_dfo_d)
        total = len(dfo_u)
        if total > 0:
            delay_rate = total_delayed / total
        else:
            delay_rate = 0
        load_per_day.append(delay_rate)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# total number of flights that arrived at PTY with arr delay > 0 / total number of flights that arrived at PTY per day
def arr_delay_rate_per_day(df_a, label, plot):
    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]

        # get flights delayed per day
        # for h in range(24):
        # h_dfd = dfd_u[dfd_u['a_Hour'] == h] # flights where PTY is the destination and arrival hour is h

        # flights delayed on arrival where PTY is destination airport
        h_dfd_d = dfd_u[dfd_u['ARR_DELAY_MINUTES'] > 0]

        # get delay rate i.e. (# flights delayed on arrival at PTY + # flights delayed on departuren from PTY) /
        # (total flights arriving at PTY + total flights leaving PTY)
        total_delayed = len(h_dfd_d)
        total = len(dfd_u)
        if total > 0:
            delay_rate = total_delayed / total
        else:
            delay_rate = 0
        load_per_day.append(delay_rate)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# sum of all departure delay times for all flights that departed from PTY per day / number of flights that arrived from PTY for that hour
def avg_dep_delay_time_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # get flights delayed per day
        # for h in range(24):
        # h_dfo = dfo_u[dfo_u['d_Hour'] == h] # flights where PTY is origin and departure hour is h

        # flights delayed on departure where PTY is origin airport
        # h_dfo_d = dfo_u[dfo_u['DEP_DELAY_MINUTES'] > 0]
        delay_time = list()
        for index, row in dfo_u.iterrows():
            delay_time.append(row['DEP_DELAY_MINUTES'])

        if len(delay_time) > 0:
            avg_delay = mean(delay_time)
        else:
            avg_delay = 0
        load_per_day.append(avg_delay)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# sum of all arrival delay times for all flights that arrived at PTY per day / number of flights that arrived at PTY for that hour
def avg_arr_delay_time_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date']) # list of all dates corresponding to flights that departed from PTY
    days_d = pd.unique(dfd['a_Date']) # list of all dates corresponding to flights that arrived from PTY

    # combine these lists to get a list we can iterate through
    dates = np.append(days_o, days_d)
    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]
        delay_time = list()

        # flights delayed on arrival where PTY is destination airport
        h_dfd_d = dfd_u[dfd_u['ARR_DELAY_MINUTES'] > 0]
        for index, row in h_dfd_d.iterrows():
            delay_time.append(row['ARR_DELAY_MINUTES'])

        if len(delay_time) > 0:
            avg_delay = mean(delay_time)
        else:
            avg_delay = 0
        load_per_day.append(avg_delay)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# stores the info for the flight that had the largest amount of departure delay for a day
def max_dep_delay_time_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date'])
    days_d = pd.unique(dfd['a_Date'])
    for d in days_o:
        dates.append(d)
    for d in days_d:
        dates.append(d)

    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the origin and departure date is u
        dfo_u = dfo[dfo['d_Date'] == u]

        # flights delayed on departure where PTY is origin airport
        if len(dfo_u) > 0:
            max_delay_time = dfo_u.iloc[0]['DEP_DELAY_MINUTES']
            for index, row in dfo_u.iterrows():
                delay_time = row['DEP_DELAY_MINUTES']
                if delay_time > max_delay_time:
                    max_delay_time = delay_time

            load_per_day.append(max_delay_time)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day

# stores the info for the flight that had the largest amount of arrival delay for a day
def max_arr_delay_time_per_day(df_a, label, plot):

    # query for records where PTY is the origin
    dfo = df_a[df_a.ORIG_CD == 56]
    dfo = dfo.sort_values('d_Date')

    # query for recrods where PTY is the destination
    dfd = df_a[df_a.DEST_CD == 56]
    dfd = dfd.sort_values('a_Date')

    # list of unique dates for all records with pty as dest or origin
    dates = list()
    days_o = pd.unique(dfo['d_Date']) # list of all dates corresponding to flights that departed from PTY
    days_d = pd.unique(dfd['a_Date']) # list of all dates corresponding to flights that arrived from PTY

    # combine these lists to get a list we can iterate through
    dates = np.append(days_o, days_d)
    dates = pd.unique(dates)
    dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, pattern))

    load_per_day = list()

    for u in dates:

        # query for records where PTY is the destination and arrival date is u
        dfd_u = dfd[dfd['a_Date'] == u]
        # flights delayed on arrival where PTY is destination airport
        if len(dfd_u) > 0:
            max_delay_time = dfd_u.iloc[0]['ARR_DELAY_MINUTES']
            for index, row in dfd_u.iterrows():
                delay_time = row['ARR_DELAY_MINUTES']
                if delay_time > max_delay_time:
                    max_delay_time = delay_time

            load_per_day.append(max_delay_time)

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    # load_per_day = normalize(load_per_day)
    if plot == True:
        plt.plot(load_per_day, label=label)
    return load_per_day




