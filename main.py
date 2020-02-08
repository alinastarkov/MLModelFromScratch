import numpy as np
from naivesbayes import NaiveBayes
from sklearn.utils import shuffle
from logregression import LogRegression 
import random


#evaluate the model accuracy
#function takes the true label and target label as input, and output the accuracy score 
# trueLabels and predictLabels are both np array
def evaluate_acc(trueLabels, predictLabels):
	#accuracy = number of correct predictions/ total number of predictions made
	correctPredictions = (trueLabels == predictLabels)
	return correctPredictions.sum() / correctPredictions.size()

#do the cross validation
def k_cross_validation(trainningData, model):
	#need to split trainning data into X and y somehow
	accuracies = []
	k_folds, y_folds = split_data(trainningData)
	for i in range(5):
		if(model=="nb"):
			nbModel=NaiveBayes() 
			nbModel.fit(k_folds[i], y_folds[i])
			testSet = k_folds[:i]+k_folds[i+1:]
			predictions = nbModel.predict(testSet)
			test_y = predictions[-1]
			accuracies.append(evaluate_acc(test_y, predictions))

		else:
			logModel=LogRegression() 
			w = logModel.fit(k_folds[i], y_folds[i], 0.001, 0.001)
			testSet = k_folds[:i]+k_folds[i+1:]
			predictions = logModel.predict(testSet, w)
			test_y = predictions[-1]
			accuracies.append(evaluate_acc(test_y, predictions))
			
	avg_accuracy = sum(accuracies)/5
	return avg_accuracy


def split_data(X):
	split_data = []
	split_y=[]
	copy_data = X
	size_folds = int(len(X)/5)

	for _ in range(5):
		fold = []
		while len(fold) < size_folds:
			i = random.randrange(len(copy_data))
			fold.append(copy_data[i])
		split_data.append(fold)
		split_y.append(fold[:,-1])
	return np.array(split_data), np.array(split_y)