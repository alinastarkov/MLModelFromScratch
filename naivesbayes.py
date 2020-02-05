import numpy has np 

class naivesBayes:
	def __init__(self):
		


	# function fit takes the training Data, hyperparameters as input
	# x = NxD array where N is the number of instances and D is the number of features
	# y = number of classes 
	def fit(X, y, self):
		num_instance = X.shape[0]
		num_features = X.shape[1]
		self.uniqueClasses = np.unique(y)
		num_classes = len(self.uniqueClasses)

		#initialize the array of the mean, standard deviation, and prior probability fill with 0
		self.priorArray = np.zeros(num_classes)
		self.meanArray, self.varArray = np.zeros(num_instance, num_features)

		# fill the mean Array, prior Array, and standard deviation array
		for uniqueClass in self.uniqueClasses:
			#select all the rows with the label, assuming that the label is always at the end 
			selectedRows = X[np.where(X[:,-1] == uniqueClass)]
			self.priorArray[uniqueClass] = selectedRows.shape[0] / num_instance
			self.meanArray[uniqueClass, :] = selectedRows.mean(axis=0) #calculate the mean across the column of the selected rows 
			self.varArray[uniqueClass, :] = selectedRows.var(axis=0)


	#take a set of input points as input and output predictions.
	#convert probabilities to binary 0-1 prediction by thresholding the ouput at 0.5
	def predictBernoulli(instances, self):
		prediction = []
		for instance in instances:
			# calculate the posterier probability y = max log(P(x1|y)) + log(P(x2|y)) + .. + log(P(xn|y))
			posterier_prob = []
			for i in range(num_classes):
				uniqueClass=self.uniqueClass[i]
				var = self.varArray[i]
				mean = self.stdArray[i]
				likelihood_term = np.sum(np.log((np.exp(-(uniqueClass-mean)**2/2*stdv))/( np.sqrt(stdv*2*np.pi))))
				posterier = likelihood_term + self.priorArray[i]
				posterier_prob.append(posterier)
			output_probability = np.argmax(posterier_prob)
			if output_probability > 0.5:
				output_prediction = 1
			elif output_prediction < 0.5:
				output_prediction = 0
			prediction.append(output_prediction)
		return np.array(prediction)


