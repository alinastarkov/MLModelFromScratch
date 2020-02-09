# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

abHeaders = ['sex', 'length', 'diameter', 'height', 'wholeWeight', 'shuckedWeight', 'visceraWeight', 'shellWeight', 'rings']

dfAb = pd.read_csv('abalone.data', sep=",", header=None, names=abHeaders, na_values=[" ?"])

#select categorical features to one hot encode
#this is only the gender feature
cat_dfAb = dfAb.select_dtypes(include=[object])
#print(cat_dfAb.head(30))

le = preprocessing.LabelEncoder()
cat2_dfAb = cat_dfAb.apply(le.fit_transform)
#print(cat2_dfHab.head(10))

enc = preprocessing.OneHotEncoder()
enc.fit(cat2_dfAb)
onehotlabels = enc.transform(cat2_dfAb).toarray()
#print(onehotlabels)
nocat_dfAb =  dfAb.select_dtypes(exclude=[object])

#now lets remove the related terms: length
nocat_dfAb=nocat_dfAb.drop(columns=['diameter', 'shuckedWeight'])
nocatlabels = nocat_dfAb.to_numpy()

allabdata= np.concatenate((onehotlabels, nocatlabels), axis=1)

X_abalone = allabdata[:, :-1]
y_abalone= allabdata[:, -1]
#print(allabdata.shape)
#print(allabdata)
#print(X_ab)
#print(y_ab)

 
#GRAPHS
#bar graph by sex
dfAb["sex"].value_counts().plot(kind='bar')
#plt.show()

#bar graph length
out = pd.cut(dfAb['length'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
#plt.show()

#bar graph diameter
out = pd.cut(dfAb['diameter'], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
#plt.show()

#bar graph height
out = pd.cut(dfAb['height'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
plt.title("Height")
#plt.show()

#bar graph wholeWeight
out = pd.cut(dfAb['wholeWeight'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
plt.title("Whole Weight")
#plt.show()

#bar graph shuckedWeight
out = pd.cut(dfAb['shuckedWeight'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
plt.title("Shucked Weight")
#plt.show()

#bar graph visceraWeight
out = pd.cut(dfAb['visceraWeight'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
plt.title("Visera Weight")
#plt.show()

#bar graph shellWeight
out = pd.cut(dfAb['shellWeight'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
plt.title("Shell Weight")
#plt.show()

#bar graph rings
dfAb["rings"].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title("Rings")
#plt.show()

#pairwise scatter
#sns.pairplot(dfAb)
#plt.savefig('ab.png')


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









