import numpy as np
import naivesbayes as nb
from sklearn.utils import shuffle
import logregression as log 


#evaluate the model accuracy
#function takes the true label and target label as input, and output the accuracy score 
# trueLabels and predictLabels are both np array
def evaluate_acc(trueLabels, predictLabels):
	#accuracy = number of correct predictions/ total number of predictions made
	correctPredictions = (trueLabels == predictLabels)
	return correctPredictions.sum() / correct.size()

#do the cross validation
def k_cross_validation(X, y, model):
	#need to split trainning data into X and y somehow
	accuracies = []
	k_folds = split_data(X)
	for i in range(5):
		if(model=="nb"):
			#nbModel=nb() fill this with attributes
			nbModel.fit(k_folds[i], y)
			testSet = k_folds[:i]+k_folds[i+1:]
			predictions = nbModel.predict(testSet)
			accuracies.append(evaluate_acc(y, predictions))

		else:
			#logModel=nb() fill this with attributes
			logModel.fit(k_folds[i])
			testSet = k_folds[:i]+k_folds[i+1:]
			predictions = nb.predict(testSet)
			accuracies.append(evaluate_acc(y, predictions))
			
	avg_accuracy = sum(accuracies)/5
	return avg_accuracy


def split_data(X):
	split_data = []
	copy_data = X
	size_folds = int(len(X)/5)

	for j in range(5):
		fold = []
		while len(fold) < size_folds:
			i = randrange(len(copy_data))
			fold.append(copy_data[i])
		split_data.append(fold)
	return np.array(split_data)
	