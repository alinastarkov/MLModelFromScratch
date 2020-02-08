import numpy as np
from naivesbayes import NaiveBayes
from sklearn.utils import shuffle
from logregression import LogRegression 
import random
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


#evaluate the model accuracy
#function takes the true label and target label as input, and output the accuracy score 
# trueLabels and predictLabels are both np array
def evaluate_acc(trueLabels, predictLabels):
	#accuracy = number of correct predictions/ total number of predictions made
	return np.sum(trueLabels == predictLabels) / len(trueLabels)

#do the cross validation
def k_cross_validation(trainningData, label, model):
	#need to split trainning data into X and y somehow
	accuracies = []
	k_folds= split_data(trainningData, label)
	for i in range(5):
		if(model=="nb"):
			Xtrain = k_folds[0].copy()
			Ytrain = k_folds[1].copy()
			testSet = k_folds[0][i]
			test_y = k_folds[1][i]
			np.delete(Xtrain, i, 0)
			np.delete(Ytrain, i, 0)
			Xtrain = np.concatenate(Xtrain)
			Ytrain = np.concatenate(Ytrain)

			nbModel=NaiveBayes() 
			nbModel.fit(Xtrain, Ytrain)
			predictions = nbModel.predict(testSet)
			accuracies.append(evaluate_acc(test_y, predictions))

	avg_accuracy = sum(accuracies)/5
	return avg_accuracy


def split_data(X, y):
	return np.array_split(X, 5), np.array_split(y, 5)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# nbModel=NaiveBayes() 
# nbModel.fit(X_train, y_train)
# predictions = nbModel.predict(X_test)
# a = evaluate_acc(y_test, predictions)
# print(a)

bc = datasets.load_iris()
X, y = bc.data, bc.target
test = k_cross_validation(X, y, "nb")
print(test)
