from sklearn.feature_selection import VarianceThreshold
from numpy import unique
import pandas as pd
from os.path import join
from flight_analysis import paths
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np

''' 
Abbey Centers
This program prepares the original data set for model training. Run as is to create the Copa_Airlines_Cleaned.csv
file. Uncomment lines 155-156 to normalize the set and create the Cleaned_Normalized.csv file.

The following changes are made to the FSU_Fully_Cleaned.csv file
1.	Enumerate string values
2.	Remove columns with single value
3.	Removed some outlier rows with negative date time (only 3) 
4.	Normalized values to lie within 0 and 1 to prevent bias
5.	Drop rows where data is missing
6.	Split time columns for individual attributes

For data cleaning and class imbalance
https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/
https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
'''


def create_dataset(normalize):
    ''' ASSIGN FNAME WITH PATH TO FSU_FULLY_CLEANED.CSV'''
    fname = paths.fsu_fully_cleaned
    df = pd.read_csv(fname)

    # drop rows where time off or on are < 0... must be an error in the data set
    df_test1 = df[df.OFF_DTMZ < 0]
    df_test2 = df[df.ON_DTMZ < 0]
    df = df.drop(df_test1.index.values)
    df = df.drop(df_test2.index.values)

    # combine scheduled date and time columns into single column and convert to datetime object
    df['sch_dep_time'] = pd.to_datetime(df['SCH_DEP_DTZ'] + ' ' + df['SCH_DEP_TMZ'])
    df['sch_arr_time'] = pd.to_datetime(df['SCH_ARR_DTZ'] + ' ' + df['SCH_ARR_TMZ'])

    # calculate the actual departure and arrival time by adding the departure and arrival delay to the scheduled times
    df['act_dep_time'] = pd.to_datetime(df['sch_dep_time']) + pd.to_timedelta(df['DEP_DELAY_MINUTES'], unit='m')
    df['act_arr_time'] = pd.to_datetime(df['sch_arr_time']) + pd.to_timedelta(df['ARR_DELAY_MINUTES'], unit='m')

    adt_cp = df['act_dep_time']
    aat_cp = df['act_arr_time']
    sdt_cp = df['sch_dep_time']
    sat_cp = df['sch_arr_time']

    # split actual and scheduled departure and arrival time columns into many other columns line hour, minute, and second
    df['d_Hour'] = pd.to_datetime(df['act_dep_time']).dt.hour
    df['d_Minute'] = pd.to_datetime(df['act_dep_time']).dt.minute
    df['d_Weekday'] = pd.to_datetime(df['act_dep_time']).dt.weekday
    df['d_Month'] = pd.to_datetime(df['act_dep_time']).dt.month
    df['d_Day'] = pd.to_datetime(df['act_dep_time']).dt.day

    df['a_Hour'] = pd.to_datetime(df['act_arr_time']).dt.hour
    df['a_Minute'] = pd.to_datetime(df['act_arr_time']).dt.minute
    df['a_Weekday'] = pd.to_datetime(df['act_arr_time']).dt.weekday
    df['a_Month'] = pd.to_datetime(df['act_arr_time']).dt.month
    df['a_Day'] = pd.to_datetime(df['act_arr_time']).dt.day

    df['sch_d_Hour'] = pd.to_datetime(df['sch_dep_time']).dt.hour
    df['sch_d_Minute'] = pd.to_datetime(df['sch_dep_time']).dt.minute
    df['sch_d_Weekday'] = pd.to_datetime(df['sch_dep_time']).dt.weekday
    df['sch_d_Month'] = pd.to_datetime(df['sch_dep_time']).dt.month
    df['sch_d_Day'] = pd.to_datetime(df['sch_dep_time']).dt.day

    df['sch_a_Hour'] = pd.to_datetime(df['sch_arr_time']).dt.hour
    df['sch_a_Minute'] = pd.to_datetime(df['sch_arr_time']).dt.minute
    df['sch_a_Weekday'] = pd.to_datetime(df['sch_arr_time']).dt.weekday
    df['sch_a_Month'] = pd.to_datetime(df['sch_arr_time']).dt.month
    df['sch_a_Day'] = pd.to_datetime(df['sch_arr_time']).dt.day

    # for any column with strings, get list of unique values and replace them with unique ID number corresponding to that label
    corr_vals = list()
    labels = list()
    consider = list()
    unique_vals = list()

    # map column names to their types and use map later to determine if a column has a string in it and if so enumerate the column
    df_types = df.dtypes  # list that stores the type for each feature (str, int, etc.)
    type_map = {}

    # map the name of a feature to it's type
    for i, v in df_types.items():
        type_map[i] = str(v)

    # iterate through each feature and it's data (loop through the columns one at a time)
    for (columnName, columnData) in df.iteritems():

        tmp_list = list()
        id = 0

        # if column data contains strings, get list of unique values in that column
        if type(columnData[0]) == str:
            # if non numeric (so strings in the column are possible) but also does not contain time data, then enumerate labels
            if type_map[columnName] == 'object' and columnData[0].find("/") == -1 and columnData[0].find(":") == -1 and \
                    columnData[0].find("-") == -1:
                tmp_map = {}  # for mapping unique strings to a numeric value

                # get number of unique items in the column
                unique_list = unique(columnData)
                unique_items = len(unique_list)

                # if there is more than one unique item in the list
                for u in unique_list:
                    tmp_list.append(u)
                    tmp_map[u] = id
                    id += 1

                # replace the actual contents of the data frame with the appropriate numeric values for the column
                df = df.replace({columnName: tmp_map})

    ''' REMOVE COLUMNS THAT COULD NOT BE ENUMERATED OR ARE STILL A STRING TYPE'''
    df_types = df.dtypes
    drop_cols = list()
    for i, v in df_types.items():
        if str(v) == 'object' or str(v) == 'datetime64[ns]':
            df = df.drop(columns=i)


    ''' DROP COLUMNS WITH ONLY 1 UNIQUE VALUE'''
    df = variance_threshold_selector(df, 0.0)

    ''' DROP ROWS WITH MISSING VALUES'''
    # list columns with missing values
    for c in df.columns[df.isnull().any()]:
        num_nans = df[c].isna().sum()
        if num_nans >= 7489:
            df = df.drop(columns=c)

    df = df.dropna(axis=0)

    newfilename = "Copa_Airlines_Cleaned.csv"

    ''' NORMALIZE VALUES TO BE BETWEEN 0 AND 1'''
    if normalize:
        df["Departure_Delay"] = df["DEP_DELAY_MINUTES"]
        df["Departure_Delay"] = np.where(df["Departure_Delay"] > 0, 1, df["Departure_Delay"])
        df["Departure_Delay"] = np.where(df["Departure_Delay"] <= 0, 0, df["Departure_Delay"])

        df = (df - df.min()) / (df.max() - df.min())
        newfilename = "Cleaned_Normalized.csv"



    df['d_Date'] = adt_cp.dt.date
    df['d_Time'] = adt_cp.dt.time
    df['a_Date'] = aat_cp.dt.date
    df['a_Time'] = aat_cp.dt.time

    ''' SAVE CLEANED DATA SET TO A NEW FILE '''
    df.to_csv(join(paths.working_dir, newfilename), index=False)


''' Remove columns from data if the variance of that column is < threshold'''
def variance_threshold_selector(data, threshold):
    features = data.columns.values.tolist()
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    fit_data = data[data.columns[selector.get_support(indices=True)]]
    new_features = fit_data.columns.values.tolist()

    removed = list()
    for o in features:
        if o not in new_features:
            removed.append(o)
    return fit_data


