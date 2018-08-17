# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:22:48 2018

@author: Pabs
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('GlassIdentification.xlsx');




X=dataset.drop(["ID","Type","Target_Names"],axis=1)

y=dataset["Type"]

#splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




"""knn"""

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

y_pred_knn = classifier_knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

from sklearn.metrics import accuracy_score

acc_knn=accuracy_score(y_test,y_pred_knn)


"""logistic regression"""
from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression(random_state = 0)
classifier_log.fit(X_train, y_train)

y_pred_log = classifier_log.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred_log)

acc_log=accuracy_score(y_test,y_pred_log)


'''Support vector machine'''
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)

y_pred_svm = classifier_svm.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)

acc_svm=accuracy_score(y_test,y_pred_svm)

'''KernelSVM'(kernel==rbf)'''
from sklearn.svm import SVC
classifier_kernelsvm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernelsvm.fit(X_train, y_train)

y_pred_kernelsvm = classifier_kernelsvm.predict(X_test)

cm_kernelsvm = confusion_matrix(y_test, y_pred_kernelsvm)

acc_kernelsvm=accuracy_score(y_test,y_pred_kernelsvm)


'''naive bayes(GaussianNB)'''
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(X_train, y_train)

y_pred_naive = classifier_naive.predict(X_test)

cm_naive = confusion_matrix(y_test, y_pred_naive)

acc_naive=accuracy_score(y_test,y_pred_naive)

'''Decision tree classifivation'''
from sklearn.tree import DecisionTreeClassifier
classifier_DecTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DecTree.fit(X_train, y_train)

y_pred_DecTree= classifier_DecTree.predict(X_test)

cm_DecTree= confusion_matrix(y_test, y_pred_DecTree)

acc_DecTree=accuracy_score(y_test,y_pred_DecTree)

'''Random forest classification'''
from sklearn.ensemble import RandomForestClassifier
classifier_RandForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RandForest.fit(X_train, y_train)

y_pred_RandForest = classifier_RandForest.predict(X_test)

cm_RandForest= confusion_matrix(y_test, y_pred_RandForest)

acc_RandForest=accuracy_score(y_test,y_pred_RandForest)


#accuracies
print("Accuracy of DecisionTree is {}\nAccuracy of Random Forest is {}\nAccuracy of KernelSVM is {}\nAccuracy of KNN is {}\nAccuracy of Logistic regression is {}\nAccuracy of Naive Bayes is {}\nAccuracy of Supprt Vector Machine(linear kernel) is {}".format(acc_DecTree,acc_RandForest,acc_kernelsvm,acc_knn,acc_log,acc_naive,acc_svm))