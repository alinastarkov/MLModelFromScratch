from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from logregression import LogRegression

iris = datasets.load_iris()

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#print(X_train.shape)

lr = LogRegression()
lr.fit(X_train, y_train, 0.01, .001)
predictions = lr.predict(X_test)
print("predictions")
print(predictions)

print("LogRegression classification accuracy", accuracy(y_test, predictions))
