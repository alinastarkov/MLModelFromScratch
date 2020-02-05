import numpy as np
import naivesbayes as nb
from sklearn.utils import shuffle 

#evaluate the model accuracy
#function takes the true label and target label as input, and output the accuracy score 
# trueLabels and predictLabels are both np array
def evaluate_acc(trueLabels, predictLabels):
	#accuracy = number of correct predictions/ total number of predictions made
	correctPredictions = (trueLabels == predictLabels)
	return correctPredictions.sum() / correct.size()

#do the cross validation
def k_cross_validation():
	