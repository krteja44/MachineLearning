#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features, labels)
pred = clf.predict(features)
print(accuracy_score(labels, pred))


from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state = 42)
clf2 = DecisionTreeClassifier()
clf2.fit(X_train, y_train)

print(accuracy_score(y_test,clf2.predict(X_test)))

""""""""""""""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

param_grid = {'criterion': ['gini', 'entropy'], 'max_depth' : list(range(10,20)), 
                   'max_leaf_nodes': list(range(2, 100))}
clf = GridSearchCV(DecisionTreeClassifier(random_state= 42), param_grid)
clf.fit(X_train, y_train)

clf.best_estimator_
pred = clf.predict(X_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
