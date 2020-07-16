#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 

from tester import dump_classifier_and_data, test_classifier
from operator import itemgetter

#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
feature_list = list(data_dict['BANNANTINE JAMES M'].keys())
feature_list.remove('poi')
feature_list.remove('email_address')
feature_list.remove('other')

### Remove columns with more than 50% Null Values
df = pd.DataFrame(data_dict).T
df.replace(to_replace = 'NaN', value = np.nan, inplace=True)
features_to_remove = []
for key in feature_list:
    if df[key].isnull().sum() > df.shape[0]*0.5:
        features_to_remove.append(key)
 
feature_list = [key for key in feature_list if key not in features_to_remove]
feature_list = ['poi'] + feature_list 

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 3: Create new feature(s)
df_temp = df[['to_messages','from_this_person_to_poi','from_messages','from_poi_to_this_person']]
### from the df_temp it is clear that we have null values for same indices for all the columns

for person in my_dataset:
    if my_dataset[person]['to_messages'] != 'NaN':
        my_dataset[person]['percent_of_messages_sent_to_poi'] =  float(my_dataset[person]['from_this_person_to_poi'])/float(my_dataset[person]['to_messages']) * 100
        my_dataset[person]['percent_of_messages_received_from_poi'] =  float(my_dataset[person]['from_poi_to_this_person'])/float(my_dataset[person]['from_messages']) * 100
    else:
        my_dataset[person]['percent_of_messages_sent_to_poi'] = 'NaN'
        my_dataset[person]['percent_of_messages_received_from_poi'] = 'NaN'
        
features_list += ['percent_of_messages_sent_to_poi', 'percent_of_messages_received_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
def select_tune_clf(n_features, features, labels):
    
    clf_list = []
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    
    ###DecisionTree
    dt = DecisionTreeClassifier()
    parameters = {'min_samples_split': list(range(2,8)),
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy']}
    clf_dt = GridSearchCV(dt, parameters, scoring='f1')
    
    
    ###RandomForest
    rf = RandomForestClassifier()
    parameters = {'min_samples_split': list(range(2,8)),
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy'],
                  'n_estimators': list(range(2,8))}
    clf_rf = GridSearchCV(rf, parameters, scoring='f1')
    
    
    ###KNearestNeighbor
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': list(range(1,10,2))}
    clf_knn = GridSearchCV(knn, parameters, scoring='f1')
    
    
    ###SupportVectorClassifier
    svc = SVC()
    parameters = {'kernel': ['rbf'],
                  'C': [1, 10, 100, 1000, 10000, 100000]}
    clf_svc = GridSearchCV(svc, parameters, scoring='f1')
    
    
    clf_dt.fit(features_train,labels_train)
    clf_dt = clf_dt.best_estimator_
    pred_dt = clf_dt.predict(features_test)
    clf_list.append([n_features, scores, clf_dt, f1_score(labels_test,pred_dt), accuracy_score(labels_test,pred_dt)])
    
    
    clf_rf.fit(features_train,labels_train)
    clf_rf = clf_rf.best_estimator_
    pred_rf = clf_rf.predict(features_test)
    clf_list.append([n_features, scores, clf_rf, f1_score(labels_test,pred_rf), accuracy_score(labels_test,pred_rf)])
    
    
    clf_knn.fit(features_train,labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    clf_list.append([n_features, scores, clf_knn, f1_score(labels_test,pred_knn), accuracy_score(labels_test,pred_knn)])
    
    
    clf_svc.fit(features_train,labels_train)
    clf_svc = clf_svc.best_estimator_
    pred_svc = clf_svc.predict(features_test)
    clf_list.append([n_features, scores, clf_svc, f1_score(labels_test,pred_svc), accuracy_score(labels_test,pred_svc)])
    
    
    sorted_clf_list = sorted(clf_list, key=lambda x: x[3])
    return sorted_clf_list[::-1][0]
    
clf_list = []
for k in range(1, len(feature_list)//2):
    clf_list.append(select_tune_clf(k, features, labels))
    
sorted_clf_list = sorted(clf_list, key=itemgetter(3,4), reverse=True)

clf = sorted_clf_list[0][2]
print('\nClf: ',clf)

number_of_features = sorted_clf_list[0][0]
print('\nNumber Of Features: ',number_of_features)

scores_list = sorted_clf_list[0][1]
feature_score = []
for i, feature in enumerate(feature_list[1:]):
    feature_score.append([feature, scores_list[i]])
feature_score = sorted(feature_score, key=itemgetter(1), reverse=True)
print('Features and Scores:\n ',feature_score)

new_features_list = []
for feature in feature_score[:number_of_features]:
    new_features_list.append(feature[0])
print('Features Used: ', new_features_list)
new_features_list = ['poi'] + new_features_list

print('F1 Score: ',sorted_clf_list[0][3],' Accuracy :',sorted_clf_list[0][4])
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, new_features_list)
test_classifier(clf, my_dataset, new_features_list)
