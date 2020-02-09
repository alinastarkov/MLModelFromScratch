import numpy as np 

class NaiveBayes:
    def __init__(self):
        pass

    def _gaussian_prob(self, x, mean, std):
        top = np.exp(- ((x-mean)**2 / (2 * std**2)))
        bottom = np.sqrt(2*np.pi* std**2)
        return np.log(top/bottom)

    def fit(self, X, y):
        self.uniqueClasses = np.unique(y)
        seperated = []

        #seperated the data by class
        for c in self.uniqueClasses: 
            #select all the rows with the label, assuming that the label is always at the end 
            selectedRows = X[y == c]
            seperated.append(selectedRows)
        
        #calculate mean, standard deviation
        summaries= np.array([(np.mean(i, axis=0), np.std(i, axis=0)) for i in seperated])
        self.summaries = []
        for subArray in summaries: 
            # the original array is like [[[mean1class1, mean2class1],[std1class1, std2]][]] we want to put it into [[[mean1class1, std1class1], [mean2, std2]]]
            summary = list(zip(*subArray))
            self.summaries.append((summary))

        self.summaries = np.array(self.summaries)

    def _get_prob(self, x):
        col_prediction = []
        #loop through the class 
        for i in self.summaries:
            class_attributes = zip(i,x)
            # for each class, calculate the prob with the gaussian function, sum the prob of the attribute
            col_prediction.append(np.sum(self._gaussian_prob(val, rowsummary[0], rowsummary[1]) for rowsummary, val in class_attributes))
        return col_prediction


    def predict(self,X):
        prediction = []
        for x in X:
            #get individual probability for each attribute for each class in each row, 
            prob_prediction = self._get_prob(x)
            prediction.append(prob_prediction)
        return np.argmax(prediction, axis=1)