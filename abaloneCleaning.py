# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

abHeaders = ['sex', 'length', 'diameter', 'height', 'wholeWeight', 'shuckedWeight', 'visceraWeight', 'shellWeight', 'rings']

dfAb = pd.read_csv('abalone.data', sep=",", header=None, names=abHeaders, na_values=[" ?"])

#select categorical features to one hot encode
cat_dfAb = dfAb.select_dtypes(include=[object])
#print(cat_dfAb.head(30))

le = preprocessing.LabelEncoder()
cat2_dfAb = cat_dfAb.apply(le.fit_transform)
#print(cat2_dfHab.head(10))

enc = preprocessing.OneHotEncoder()
enc.fit(cat2_dfAb)
onehotlabels = enc.transform(cat2_dfAb).toarray()
print(onehotlabels)
nocat_dfAb =  dfAb.select_dtypes(exclude=[object])
nocatlabels = nocat_dfAb.to_numpy()



allabdata= np.concatenate((onehotlabels, nocatlabels), axis=1)
#allabdata = allhabdata[:,:-1]

#dropping the last column so if survived more than 5 years->1 and survived less than 5 years is 0
#print(allabdata)

 
#GRAPHS
#bar graph age
#out = pd.cut(dfHab['age'], bins=[0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], include_lowest=True)
#ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
#plt.show()


#bar graph opyear
#dfHab['opYear'].value_counts().sort_index(ascending=False).plot(kind='bar')
#plt.show()

#bar graph nodes
#dfHab['nodes'].value_counts().plot(kind='bar')
#plt.show()

#bar graph class
#dfHab['survival'].value_counts().plot(kind='bar')
#plt.show()


#scatter class vs age
#this is the one that worked
"""
g = sns.FacetGrid(dfHab, col="survival")
g = g.map(plt.hist, "age")
plt.show()

#scatter class vs opyear
g = sns.FacetGrid(dfHab, col="survival")
g = g.map(plt.hist, "opYear")
plt.show()

#scatter class vs nodes
g = sns.FacetGrid(dfHab, col="survival")
g = g.map(plt.hist, "nodes")
plt.show()

#scatter age vs nodes
dfHab.plot(kind='scatter',x='age',y='nodes',color='blue')
plt.show()

#scatter Opyear vs nodes
dfHab.plot(kind='scatter',x='opYear',y='nodes',color='blue')
plt.show()

sns.pairplot(dfHab)
plt.savefig('a.png')
"""
#scatter age vs opyear


#notes
"""
dfHab.plot(kind='bar',x='opYear',y='survival',color='blue')
plt.show()

dfHab.plot(kind='bar',x='nodes',y='survival',color='green')
plt.show()

dfHab.plot(kind='bar',x='opYear',y='nodes',color='green')
plt.show()

dfHab['age'].value_counts().plot(kind='bar')
plt.show()
dfHab=dfHab.sort_values(by='opYear')

df = dfHab.groupby("survival")
df['age'].value_counts().plot(kind='scatter')
plt.show()
"""



#out = pd.cut(dfHab.groupby('survival')['age'], bins=[0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], include_lowest=True)
#dfHab.groupby('survival')['age'].value_counts().sort_index().plot.bar(by='survival')
#ax = out.value_counts().plot(kind='bar')
#plt.show()

#dfHab['age'].value_counts().sort_index().plot.bar(by='survival', subplots=True)

#ax = out.value_counts().plot(kind='bar')
#plt.show()

#avg_age_per_year = dfHab.groupby('opYear')['age'].mean() 
#dfHab.plot(kind='scatter',x='opYear',y='avg_age_per_year',color='green')
#plt.show()






