#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


data_temp_df = pd.DataFrame(data_dict)
data_temp_df = data_temp_df.T
data_temp_df[data_temp_df.salary != 'NaN'].sort_values(by = 'salary', ascending = False)
data_temp_df.drop(labels='TOTAL', inplace = True)
bandits = data_temp_df[data_temp_df.salary != 'NaN'].sort_values(by = 'salary', ascending = False)
data_temp_df[data_temp_df.salary > 1000000]
