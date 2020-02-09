import numpy as np
from sklearn.utils import shuffle
from logregression import LogRegression 
from naivesbayes import NaiveBayes
import random

#import clean data 
from ionosphere_cleaning import X_ionosphere, y_ionosphere
from HabCleaning import X_haberman, y_haberman
from cleanAdult import X_adult, y_adult, nocatlabels, onehotlabels
from abaloneCleaning import X_abalone, y_abalone

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
			# nbModel = GaussianNB()
			# nbModel.fit(Xtrain, Ytrain)
			# predictions = nbModel.predict(testSet)
			accuracies.append(evaluate_acc(test_y, predictions))

		else:
			Xtrain = k_folds[0].copy()
			Ytrain = k_folds[1].copy()
			testSet = k_folds[0][i]
			test_y = k_folds[1][i]
			np.delete(Xtrain, i, 0)
			np.delete(Ytrain, i, 0)
			Xtrain = np.concatenate(Xtrain)
			Ytrain = np.concatenate(Ytrain)

			logModel=LogRegression(0.001, 10000) 
			w = logModel.fit(Xtrain, Ytrain)
			predictions = logModel.predict(testSet, w)
			accuracies.append(evaluate_acc(test_y, predictions))

	avg_accuracy = sum(accuracies)/5
	return avg_accuracy


def split_data(X, y):
	return np.array_split(X, 5), np.array_split(y, 5)

print("haberman")
test = k_cross_validation(X_haberman, y_haberman, "nb")
testlog = k_cross_validation(X_haberman, y_haberman, "log")
print(test)
print(testlog)

print("Adult")
np.set_printoptions(threshold=np.inf)
print("Gaussian naive bayes")
test = k_cross_validation(nocatlabels, y_adult, "nb")
testlog = k_cross_validation(nocatlabels, y_adult, "log")
print(test)
print(testlog)
print("Mixed naive bayes")

print("Bernoulli naives bayes")

print("Ionosphere")
test = k_cross_validation(X_ionosphere, y_ionosphere, "nb")
testlog = k_cross_validation(X_ionosphere, y_ionosphere, "log")
print(test)
print(testlog)

