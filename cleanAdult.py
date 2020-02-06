import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

adultHeaders=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'] 

#dfIonosphere = pd.read_csv('ionosphere.txt', sep=',', header=None)
dfAdult = pd.read_csv('adult.data', sep=",", header=None, names=adultHeaders, na_values=[" ?"])
#dfRedWine = pd.read_csv('winequality-red.csv')

new_dfAdult= dfAdult.dropna(axis = 'columns', how ='any') 

#dfAdult['income'].value_counts().plot(kind='bar')
#plt.show()

#out = pd.cut(new_dfAdult['hours-per-week'], bins=[0, 20, 30, 40, 50, 60, 80, 120], include_lowest=True)
#ax = out.value_counts(sort=False).plot.bar(rot=0, figsize=(6,4))
#plt.show()

#dfSample = new_dfAdult.sample(1000) 
#xdataSample, ydataSample = new_dfAdult["education-num"], new_dfAdult["capital-gain"]
#sns.regplot(x=xdataSample, y=ydataSample) 
#plt.show()

#dfSample = new_dfAdult.sample(500) 
#sns.pairplot(new_dfAdult)
#plt.savefig('pairwise.png')

#select categorical features
cat_dfAdult = new_dfAdult.select_dtypes(include=[object])
#print(cat_dfAdult.head(30))
#print(cat_dfAdult.columns)

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()


# use df.apply() to apply le.fit_transform to all columns
cat2_dfAdult = cat_dfAdult.apply(le.fit_transform)


enc = preprocessing.OneHotEncoder()
enc.fit(cat2_dfAdult)
onehotlabels = enc.transform(cat2_dfAdult).toarray()
#print(onehotlabels.shape)
#print(cat2_dfAdult.head())
print(onehotlabels)


