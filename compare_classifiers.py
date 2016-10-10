from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# Make those CLASSIFIERS
clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = GaussianNB()
clf4 = neighbors.KNeighborsClassifier()


# Get some DATAZ for training
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']



# To train them is our cause!

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)


# WHO WE GONNA TEST?

X_test= [[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
y_test= ['male','male','male','female','female','female','male','male']


# Make predictions! YOWZA

y_prediction_1 = clf1.predict(X_test)
y_prediction_2 = clf2.predict(X_test)
y_prediction_3 = clf3.predict(X_test)
y_prediction_4 = clf4.predict(X_test)



# WHO WINS??? :D

print("Prediction for DecisionTreeClassifier : {}\n Accuracy for DecisionTreeClassifier : {}\n".format(y_prediction_1, accuracy_score(y_test, y_prediction_1)))
print("Prediction for SupportVectorMachine : {}\n Accuracy for SupportVectorMachine : {}\n".format(y_prediction_2, accuracy_score(y_test, y_prediction_2)))
print("Prediction for Naive Bayes : {}\n Accuracy for Naive Bayes : {}\n".format(y_prediction_3, accuracy_score(y_test, y_prediction_3)))
print("Prediction for KNeighborsClassifier : {}\n Accuracy for KNeighborsClassifier : {}\n".format(y_prediction_4, accuracy_score(y_test, y_prediction_4)))

