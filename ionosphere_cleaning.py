# -*- coding: utf-8 -*-
"""

@author: Dylan Caverly

Note: 'ionosphere.data' must be renamed to 'ionosphere.csv' before 
      running this script.
      
"""

import numpy as np
import csv
from scipy import stats

# Here, we load the data and save it as a numpy array
with open('ionosphere.csv', 'r') as f: 
    all_data = list(csv.reader(f, delimiter=','))
io_all = np.array(all_data[:])

# Here, we devide the data into the features (X) and the results (Y)
io_Y = io_all[:,-1]
io_X = io_all[:,:-1]


# Here, we convert the results (good, bad) into logistic values (g = 1, b= 0)
for i in range(len(io_Y)): 
    if io_Y[i] == 'g':
        io_Y[i] = 1
    elif io_Y[i] =='b':
        io_Y[i] = 0

# convert the data to an array of floats
io_all = np.array(io_all, dtype=float) 

# Here, we divide the data into the "good" and "bad" subsets. We accomplish
# this by sorting the data by the ouput column, then splitting the array
# into the section with a good output (1) and a bad ouput (0)

io_all_sorted = io_all[io_all[:,-1].argsort()]
i_bk_temp = io_all_sorted.T[-1,:]
i_bk_temp = np.array(i_bk_temp, dtype = float)
i_bk = np.sum(1-i_bk_temp)
io_all_b, io_all_g = np.vsplit(io_all_sorted,[int(i_bk)])

io_Xb = io_all_b[:,:-1]
io_Yb = io_all_b[:,-1]

io_Xg = io_all_g[:,:-1]
io_Yg = io_all_g[:,-1]


#  Here we remove the "unnecessary" features. For each feature, all instances 
# are examined. If the mode of the distribution of the feature occurs in more
# than 60% of all instances, the feature is removed.
#  We First apply this feature to the "good" data, then the "bad" data

counter = 0
for i in range(len(io_Xg.T)):
    i = i-counter
    if (np.sum(io_Xg.T[i] == stats.mode(io_Xg.T[i]))> 0.6*(len(io_Xg)) ):
        io_Xg = np.delete(io_Xg,i,axis=1)
        io_Yg = np.delete(io_Yg,i)
        counter +=1
        
counter2 = 0
for j in range(len(io_Xb.T)):
    j = j-counter2
    if (np.sum(io_Xb.T[j] == stats.mode(io_Xb.T[j]))> 0.6*(len(io_Xb)) ):
        io_Xb = np.delete(io_Xb,j,axis=1)
        io_Yb = np.delete(io_Yb,j)
        counter2 +=1


# Now, all instances with "good" results are found in io_Xg, and their 
# corresponding results are in io_Yg.
# All instances with "bad" results are found in io_Xb, and their 
# corresponding results are in io_Yb.

