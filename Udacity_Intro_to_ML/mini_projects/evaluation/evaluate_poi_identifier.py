#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
#sort_keys = '../tools/python2_lesson14_keys.pkl'


### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  *


from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state = 42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred =  clf.predict(X_test)

print(accuracy_score(y_test,clf.predict(X_test)))
print(confusion_matrix(y_test,clf.predict(X_test)))
print(clf.predict(X_test))
print(y_test)

# How many POIs are in the test set for your POI identifier?
sum(pred)

# How many people total are in your test set?
len(pred)

# If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
accuracy_score(y_test, pred)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

precision_score(true_labels,predictions )
recall_score(true_labels, predictions)
