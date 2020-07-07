#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,`
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

type(enron_data)

len(enron_data)
for key, values in enron_data.items():
    print(key)

enron_data_df = pd.DataFrame(enron_data)
enron_data_df = enron_data_df.T
enron_data_df.reset_index(inplace = True)
enron_data_df = enron_data_df.rename(columns={'index':'Name'})
len(enron_data_df[enron_data_df['poi'] == True])
enron_data_df[enron_data_df['Name']== 'prentice james'.upper()].total_stock_value 
enron_data_df[enron_data_df['Name']== 'Colwell Wesley'.upper()].from_this_person_to_poi


