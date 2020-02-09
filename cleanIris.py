# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

irisHeaders = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']

dfIris = pd.read_csv('iris.data', sep=",", header=None, names=irisHeaders, na_values=[" ?"])
#iris has no NA or categorical data
#want to make the classification binary. Only look at class setosa vs not setosa
#select categorical features to one hot encode: this will be the 3 classes
#this is only the gender feature
cat_dfIris = dfIris.select_dtypes(include=[object])
#print(cat_dfIris.head(30))

le = preprocessing.LabelEncoder()
cat2_dfIris = cat_dfIris.apply(le.fit_transform)
#print(cat2_dfIris.tail(52))

enc = preprocessing.OneHotEncoder()
enc.fit(cat2_dfIris)
onehotlabels = enc.transform(cat2_dfIris).toarray()

#dropping last two columns so label is 1 or 0
#one hot labels is the y since it is just the labels
onehotlabels= onehotlabels[:,0]
#print(onehotlabels)

#now lets handle the continuous vars
nocat_dfIris=dfIris.drop(columns=['class'])
nocatlabels = nocat_dfIris.to_numpy()

X_iris = nocatlabels
y_iris = onehotlabels

#now lets remove the related terms: length
"""
nocat_dfIris=nocat_dfIris.drop(columns=['diameter', 'shuckedWeight'])
nocatlabels = nocat_dfAb.to_numpy()
"""

#print(X_iris)
#print(y_iris)
#print(y_iris.dtype)

 
#GRAPHS
#bar graph sepal length
dfIris["sepalLength"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Sepal Length")
#plt.show()

#bar graph sepal width
dfIris["sepalWidth"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Sepal Width")
#plt.show()

#bar graph petal length
dfIris["petalLength"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Petal Length")
#plt.show()

#bar graph petal width
dfIris["petalWidth"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Petal Width")
#plt.show()

#bar graph class
dfIris["class"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Iris Class")
#plt.show()



#bar graph 
#out = pd.cut(dfAb['length'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], include_lowest=True)
#ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
#plt.show()


#plt.show()

#pairwise scatter
sns.pairplot(dfIris, hue='class')
plt.savefig('iris.png')











