import numpy as np 

class NaiveBayes:
    def __init__(self, bernoulli):
        self.bernNB = bernoulli #0 is gaussian NB, 1 is mixed bernoulli and gaussian

    def _gaussian_prob(self, x, mean, std):
        top = np.exp(- ((x-mean)**2 / (2 * std**2)))
        bottom = np.sqrt(2*np.pi* std**2)
        return np.log(top/bottom)

    def fit(self, X, y, Xcat=[]):
        # Gaussian prob
        num_instances = X.shape[0]
        self.uniqueClasses = np.unique(y)
        seperatedByClasses = []

        #seperated the data by class
        for c in self.uniqueClasses: 
            #select all the rows with the label, assuming that the label is always at the end 
            selectedRows = X[y == c]
            seperatedByClasses.append(selectedRows)
        
        #calculate mean, standard deviation
        summaries= np.array([(np.mean(i, axis=0), np.std(i, axis=0)) for i in seperatedByClasses])
        self.summaries = []
        for subArray in summaries: 
            # the original array is like [[[mean1class1, mean2class1],[std1class1, std2]][]] we want to put it into [[[mean1class1, std1class1], [mean2, std2]]]
            summary = list(zip(*subArray))
            self.summaries.append((summary))

        self.summaries = np.array(self.summaries)

        #this part is for bernoulli
        self.priors = []

        if (self.bernNB == 1 and len(Xcat) > 0):
            Xcat_features_count = []
            Xcat_instance_count = []
            for c in self.uniqueClasses: 
                #calculate the priors prob, prior = num of instances in class / total num of instances 
                selectedRows = Xcat[y == c]
                self.priors.append(len(selectedRows)/num_instances)
                # + 0.5 to prevent 0 probability 
                arr = np.sum(np.array(selectedRows), axis=0) + 0.5
                # sum up total number of 1s across all features in that class 
                Xcat_features_count.append(arr)
                # total number of instances in that class 
                arr2= np.array(len(selectedRows) + 1.5)
                Xcat_instance_count.append(arr2)

            Xcat_features_count = np.array(Xcat_features_count)
            Xcat_instance_count = np.array(Xcat_instance_count)
            self.likelihood = Xcat_features_count / Xcat_instance_count[np.newaxis].T
        
    def _get_prob(self, x):
        col_prediction = []
        #loop through the class 
        for i in self.summaries:
            class_attributes = zip(i,x)
            # for each class, calculate the prob with the gaussian function, sum the prob of the attribute
            col_prediction.append(np.sum(self._gaussian_prob(val, rowsummary[0], rowsummary[1]) for rowsummary, val in class_attributes))
        return col_prediction

    def predict(self,X, Xcat=[]):
        #prediction for gaussian
        prediction = []
        predictionBernoulli = []
        for x in X:
            #get individual probability for each attribute for each class in each row, 
            poster_prediction = self._get_prob(x)
            prediction.append(poster_prediction)

        #prediction for bernoulli
        if (self.bernNB ==1 and len(Xcat) > 0):
            predictionBernoulli = []
            for x in Xcat:
                logp = np.log(self.priors) + (np.log(self.likelihood) * x + np.log(1 - self.likelihood) * np.abs(x - 1)).sum(axis=1) 
                logp -= np.max(logp)
                posterior = np.exp(logp)
                posterior /= np.sum(posterior)
                predictionBernoulli.append(posterior)
            hybrid = []
            for bernoulli, gaussian in zip(predictionBernoulli, prediction):
                prob = bernoulli + gaussian
                hybrid.append(prob)
            hybrid = np.array(hybrid)
            return np.argmax(hybrid, axis=1)

        return np.argmax(prediction, axis=1)
       
            