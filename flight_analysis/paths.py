
import os
from os.path import join


# get file path names for each dataset used for analysis
working_dir = os.getcwd()

# get parent directory
# working_dir = os.path.join(working_dir, '..')

plot_dir = join(working_dir, 'TESTRUN')
copa_per_day_file = join(working_dir, 'Copa_Airlines_per_day.csv') # one sample per day i.e. a variation on Copa_Airlines_Cleaned.csv
copa_airlines_cleaned_file = join(working_dir, 'Copa_Airlines_Cleaned.csv') # cleaned version of original set provided by Copa Airlines, FSU_Fully_Cleaned.csv
copa_normalized_file = join(working_dir, 'Cleaned_Normalized.csv') # a variation on Copa_Airlines_Cleaned.csv but each attribute is normalized
fsu_fully_cleaned = join(working_dir, 'FSU_fully_cleaned.csv')

