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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range =   range(1,10)
#scores={}
scores_list = []
error = []
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors= k)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    scores_list.append(accuracy_score(labels_test, pred))
    error.append(np.mean( pred != labels_test))

plt.figure(figsize=(12, 6))
plt.plot(k_range, scores_list, color='black', linestyle='dashed', marker='o',
         markerfacecolor='grey', markersize=10)
plt.xlabel('Values of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(k_range, error, color='black', linestyle='dashed', marker='o',
         markerfacecolor='grey', markersize=10)
plt.xlabel('Values of K for KNN')
plt.ylabel('Testing Error')
plt.show()

clf = KNeighborsClassifier(n_neighbors=4,algorithm = 'ball_tree')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(accuracy_score(labels_test, pred))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
