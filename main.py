# ALL THE EXPERIMENT ARE COMMENTED
import numpy as np
from sklearn.utils import shuffle
from logregression import LogRegression 
from naivesbayes import NaiveBayes
import random
from sklearn.model_selection import train_test_split

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
			logModel=LogRegression(0.001, 0.01) 
			w = logModel.fit(Xtrain, Ytrain)
			predictions = logModel.predict(testSet, w)
			accuracies.append(evaluate_acc(test_y, predictions))

	avg_accuracy = sum(accuracies)/5
	return avg_accuracy


def split_data(X, y, Xcat):
	return np.array_split(X, 5), np.array_split(y, 5), np.array_split(Xcat, 5)

def validation_nb_categorical(Xtrain, Xcattrain, X_test, Xcat_test, y_train, y_test):
	accuracies = []
	for _ in range(5):
		nbModel= NaiveBayes(1) # if be.fit(1)
		nbModel.fit(Xtrain, y_train, Xcattrain)
		predictions = nbModel.predict(X_test,Xcat_test)
		pred = evaluate_acc(y_test, predictions)
		accuracies.append(pred)
	return (sum(accuracies)/5)

def validation_nb_cont(Xtrain, X_test, y_train, y_test):
	accuracies = []
	for _ in range(5):
		nbModel= NaiveBayes(0) # if be.fit(1)
		nbModel.fit(Xtrain, y_train)
		predictions = nbModel.predict(X_test)
		pred = evaluate_acc(y_test, predictions)
		accuracies.append(pred)
	return (sum(accuracies)/5)

def validation_logistic(Xtrain, Xtest, y_train, y_test):
	accuracies = []
	for _ in range(5):
		logModel=LogRegression(0.001, 0.01) 
		w = logModel.fit(Xtrain, y_train)
		predictions = logModel.predict(Xtest, w)
		accuracies.append(evaluate_acc(y_test, predictions))
	return (sum(accuracies)/5)

# --------EXPERIMENT 2----------
print("EXPERIMENT 2")
print("haberman")
X_train, X_test, y_train, y_test = train_test_split(X_haberman_cont, y_haberman, test_size=0.10, random_state=42)
X_traincat, X_testcat, y_traincat, y_testcat = train_test_split(X_haberman_cat, y_haberman, test_size=0.10, random_state=42)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_haberman, y_haberman, test_size=0.10, random_state=42)

print("naive bayes training ")
test = k_cross_validation(X_train, yh_train, "nb", 1, X_traincat)
print(test)
print("Logistic regression training")
testlog = k_cross_validation(Xh_train, yh_train, "log", 0)
print(testlog)

print("Adult")
X_train, X_test, y_train, y_test = train_test_split(X_adult_cont, y_adult, test_size=0.10, random_state=42)
X_traincat, X_testcat, y_traincat, y_testcat = train_test_split(X_adult_cat, y_adult, test_size=0.10, random_state=42)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_adult, y_adult, test_size=0.10, random_state=42)
print("naive bayes training ")
test = k_cross_validation(X_train, yh_train, "nb", 1, X_traincat)
print(test)
print("Logistic regression training")
testlog = k_cross_validation(Xh_train, yh_train, "log", 0)
print(testlog)

print("Iris")
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.10, random_state=42)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
print("naive bayes training ")
test = k_cross_validation(X_train, y_train, "nb", 0)
print(test)
print("Logistic regression training")
testlog = k_cross_validation(X_train, y_train, "log", 0)
print(testlog)

print("Ionosphere")
X_train, X_test, y_train, y_test = train_test_split(X_ionosphere, y_ionosphere, test_size=0.10, random_state=42)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
print("naive bayes training ")
test = k_cross_validation(X_train, y_train, "nb", 0)
print(test)
print("Logistic regression training")
testlog = k_cross_validation(X_train, y_train, "log", 0)
print(testlog)



#------EXPERIMENT 3---------
#~~~~~~NAIVES BAYES~~~~~~
#split x and y into 90-10 training and testing set 


print("EXPERIMENT 2- NAIVE BAYES")
X_train, X_test, y_train, y_test = train_test_split(X_haberman_cont, y_haberman, test_size=0.10, random_state=42)
X_traincat, X_testcat, y_traincat, y_testcat = train_test_split(X_haberman_cat, y_haberman, test_size=0.10, random_state=42)

nbModel= NaiveBayes(1) # if be.fit(1)
accuracies = []
for _ in range(5):
	nbModel.fit(X_train, y_train, X_traincat)
	predictions = nbModel.predict(X_test,X_testcat)
	pred = evaluate_acc(y_test, predictions)
	accuracies.append(pred)

print("test size = 100%")
print("Validation accuracy")
print(sum(accuracies)/5)
print("training accuracy")
test = k_cross_validation(X_train, y_train, "nb", 0, X_traincat)
print(test)

test_size = [0.15, 0.30, 0.45] 

for size in test_size:
	X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=size, random_state=5)
	X_traincat1, X_testcat1, y_traincat1, y_testcat1 = train_test_split(X_traincat, y_traincat, test_size=size, random_state=5)
	accuracies = []

	for _ in range(5):
		nbModel= NaiveBayes(1) # if be.fit(1)
		nbModel.fit(X_train1, y_train1, X_traincat1)
		predictions = nbModel.predict(X_test,X_testcat)
		pred = evaluate_acc(y_test, predictions)
		accuracies.append(pred)
	print("test size " + str(size))
	print("Validation accuracy")
	print(sum(accuracies)/5)
	print("Training accuracy")
	test = k_cross_validation(X_train1, y_train1, "nb", 0,X_traincat1 )
	print(test)


# ~~~~~LOGISTIC REGRESSION ~~~~~~


print("EXPERIMENT 2- LOGISTIC REGRESSION")
X_train, X_test, y_train, test_y = train_test_split(X_haberman, y_haberman, test_size=0.10)
accuracies =[]
for _ in range(5):
	logModel=LogRegression(0.001, 0.01) 
	w = logModel.fit(X_train, y_train)
	predictions = logModel.predict(X_test, w)
	accuracies.append(evaluate_acc(test_y, predictions))

print("test size = 100%")
print("Validation accuracy")
print(sum(accuracies)/5)
print("training accuracy")
test = k_cross_validation(X_train, y_train, "log", 0)
print(test)

test_size = [0.15, 0.30, 0.45] 

for size in test_size:
	X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=size)
	accuracies = []

	for _ in range(5):
		X_train, X_test, y_train, test_y = train_test_split(X_iris, y_iris, test_size=0.10)
		logModel=LogRegression(0.001, 0.01) 
		w = logModel.fit(X_train, y_train)
		predictions = logModel.predict(X_test, w)
		accuracies.append(evaluate_acc(test_y, predictions))
	print("test size " + str(size))
	print("Validation accuracy")
	print(sum(accuracies)/5)
	print("Training accuracy")
	test = k_cross_validation(X_train1, y_train1, "log", 0 )
	print(test)

# ------------- EXPERIMENT 1 -------------
print("EXPERIMENT 1")
print("Ionosphere")
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_ionosphere, y_ionosphere, test_size=0.10, random_state=42)

print("naive bayes training ")
test = k_cross_validation(Xh_train, yh_train, "nb", 0)
print(test)
print("Logistic regression training")
testlog = k_cross_validation(Xh_train, yh_train, "log", 0)
print(testlog)

print("Iris")
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_iris, y_iris, test_size=0.10, random_state=42)
print("naive bayes")
test = k_cross_validation(X_iris, y_iris, "nb", 0)
print(test)
print("Logistic regression")
testlog = k_cross_validation(X_iris, y_iris, "log", 0)
print(testlog)
