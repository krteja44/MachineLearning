# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:03:55 2020

@author: ravit
"""


#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
import numpy as np

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 50)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print(accuracy_score(labels_test, pred))
print(confusion_matrix(labels_test, pred))

l_rate = np.linspace(start = 0.1, stop= 1, num = 10, endpoint= True)
scores_list = []
error = []
for l in l_rate:
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 50, learning_rate= l)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    scores_list.append(accuracy_score(labels_test, pred))
    error.append(np.mean( pred != labels_test))
    
plt.figure(figsize=(12, 6))
plt.plot(l_rate, scores_list, color='black', linestyle='dashed', marker='o',
         markerfacecolor='grey', markersize=10)
plt.xlabel('Values of learning_rate')
plt.ylabel('Testing Accuracy')
plt.show()
    
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators = 50, learning_rate=0.4)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print(confusion_matrix(labels_test, pred))   
print(accuracy_score(labels_test, pred))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass