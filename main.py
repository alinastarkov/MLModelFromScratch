import numpy as np
from sklearn.utils import shuffle
from logregression import LogRegression 
from naivesbayes import NaiveBayes
import random

#import clean data 
from ionosphere_cleaning import X_ionosphere, y_ionosphere
from HabCleaning import X_haberman_cat, X_haberman_cont, y_haberman, X_haberman
from cleanAdult import X_adult_cat, X_adult_cont, y_adult, X_adult
from cleanIris import X_iris, y_iris

#evaluate the model accuracy
#function takes the true label and target label as input, and output the accuracy score 
# trueLabels and predictLabels are both np array
def evaluate_acc(trueLabels, predictLabels):
	#accuracy = number of correct predictions/ total number of predictions made
	return np.sum(trueLabels == predictLabels) / len(trueLabels)

#do the cross validation
def k_cross_validation(trainningData, label, model, bernoulli, Xcat=[]):
	#need to split trainning data into X and y somehow
	accuracies = []
	n_iter = []
	k_folds= split_data(trainningData, label, Xcat)
	for i in range(5):
		if(model=="nb"):
			Xtrain = k_folds[0].copy()
			Ytrain = k_folds[1].copy()
			testSet = k_folds[0][i]
			test_y = k_folds[1][i]
			Xtrain = np.delete(Xtrain, i, 0)
			Ytrain = np.delete(Ytrain, i, 0)
			Xtrain = np.concatenate(Xtrain)
			Ytrain = np.concatenate(Ytrain)

			Xcattrain = []
			Xcattest = []
			if(bernoulli == 1):
				Xcattrain = k_folds[2].copy()
				Xcattest = k_folds[2][i]
				Xcattrain = np.delete(Xcattrain, i, 0)
				Xcattrain = np.concatenate(Xcattrain)

			nbModel=NaiveBayes(bernoulli) # if bernoulli =0 only run gaussian, otherwise mix gaussian and bernoulli
			if (bernoulli == 0):
				nbModel.fit(Xtrain, Ytrain)
				predictions = nbModel.predict(testSet)
			elif (bernoulli == 1 and len(Xcat) > 0):
				nbModel.fit(Xtrain, Ytrain, Xcattrain)
				predictions = nbModel.predict(testSet, Xcattest)
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
			w, n_iter = logModel.fit(Xtrain, Ytrain)
			predictions = logModel.predict(testSet, w)
			accuracies.append(evaluate_acc(test_y, predictions))

	avg_accuracy = sum(accuracies)/5
	return avg_accuracy, n_iter


def split_data(X, y, Xcat):
	return np.array_split(X, 5), np.array_split(y, 5), np.array_split(Xcat, 5)


print("haberman")
print("naive bayes")
test, n_iter = k_cross_validation(X_haberman_cont, y_haberman, "nb", 1, X_haberman_cat)
print(test)
print("Logistic regression")
testlog, n_iter = k_cross_validation(X_haberman, y_haberman, "log", 0)
print(testlog)

print("Adult")
print("Mixed naive bayes")
test, n_iter = k_cross_validation(X_adult_cont, y_adult, "nb", 1, X_adult_cat)
print(test)
print("Logistic regression")
testlog, n_iter = k_cross_validation(X_adult, y_adult, "log", 0)
print(testlog)

print("Ionosphere")
print("naive bayes")
test, n_iter = k_cross_validation(X_ionosphere, y_ionosphere, "nb", 0)
print(test)
print("Logistic regression")
testlog = k_cross_validation(X_ionosphere, y_ionosphere, "log", 0)
print(testlog)

print("Iris")
print("naive bayes")
test, n_iter = k_cross_validation(X_iris, y_iris, "nb", 0)
print(test)
print("Logistic regression")
testlog, n_iter = k_cross_validation(X_iris, y_iris, "log", 0)
print(testlog)


