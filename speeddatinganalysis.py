"""
    Filename:           speeddatinganalysis.py
    Author:             Eustace Ednacot
    Date Created:       Wed Mar 27 14:33:12 2019
    Date last modified: 
    Python Version:     2.7
        
@author: eustacee
"""

import pandas as pd
import numpy  as np

###############################################################################
# Function Definitions
###############################################################################
def imputeKNN(dat, k):
    """
    Perform imputation of NaNs in input DataFrame dat using the KNN algorithm.
    Return updated DataFrame with NaNs replaced with the column means of the 
    KNN for the particular column the NaNs are located in
    
    :param: dat
    :type: DataFrame
    :param: k
    :type: int
    """
    from sklearn import preprocessing
    from sklearn.neighbors import NearestNeighbors
    
    # Assert input dat as type DataFrame
    assert isinstance(dat,pd.core.frame.DataFrame)
    # Assert input k as type integer
    assert isinstance(k,int)
    
    # Replace NaNs with 0's for data standardization
    dat0  = dat.copy().fillna(0)
    dat0s = pd.DataFrame(data    = preprocessing.scale(dat0),  \
                         index   = dat0.index, \
                         columns = dat0.columns)
    # Generate a Nearest Neighbors model 
    nnmodel = NearestNeighbors(n_neighbors = k, algorithm = 'brute', \
                          metric = 'euclidean')
    # Fit standardized data to model
    nnmodel.fit(dat0s)
    # Instantiate empty list to contain NaN locations
    nanloc = []
    # Append NaN locations [index, column name] to nanloc
    for index, row in dat.iterrows():
        # If there is a NaN in the current row
        if any(pd.isnull(row)):
            # Get column names wher NaN is located in current row 
            colname = dat.columns[np.argwhere(pd.isnull(row))]
            # Append list to nanloc containing index and colnames of NaNs
            nanloc.append([int(index), colname])    
    # Loop over location pairs  
    for l in nanloc:
        # Use index to use row of dataframe as testing point.
        point = [dat0s.loc[l[0], : ]]
        # Find k nearest neighbors to point. Nearest neighbor to a point is 
        # itself. Remove first element. Resulting list contains indices of the 
        # k nearest neighbors.
        nn = map(int, nnmodel.kneighbors(point, n_neighbors = k + 1, \
                                return_distance = False)[0][1:].tolist())
        # Loop thru NaN columns in row of index
        for c in l[1]:
            # Replace specific NaN location with mean of nearest neighbors.
            dat.at[l[0], c] = dat0.loc[nn, c].mean()
    return dat   

###############################################################################
# Data Prep-Processing
###############################################################################
# Convert csv file into DataFrame format.
filename = 'Speed Dating Data.csv'
df = pd.read_csv(filename)

# Remove certain feature columns from dataframe using the data key.
col_rmv = ['zipcode', 'from', 'mn_sat', 'undergra', 'career', 'condtn', \
           'field', 'length']
df = df.drop(columns = col_rmv)
# Remove mid-experiment survey answers. Few subjects answered these questions.
col_rmv = df.columns[range(df.columns.get_loc("attr1_s"), \
                          df.columns.get_loc("amb3_s")+1)]
df = df.drop(columns = col_rmv)

# Separate columns of survey answers from timeframe 2 and 3 from the dataset.
# Timeframe 2
col_t2 = df.columns[range(df.columns.get_loc("satis_2"), \
                          df.columns.get_loc("amb5_2")+1)]
df2 = df[col_t2].copy()
df = df.drop(columns = col_t2)
# Timeframe 3
col_t3 = df.columns[range(df.columns.get_loc("you_call"), \
                          df.columns.get_loc("amb5_3")+1)]
df3 = df[col_t3].copy()
df = df.drop(columns = col_t3)



# Get column index values from 
r1_1   = range(df.columns.get_loc("attr1_1"),df.columns.get_loc("shar1_1")+1)
r2_1   = range(df.columns.get_loc("attr2_1"),df.columns.get_loc("shar2_1")+1)
r3_1   = range(df.columns.get_loc("attr3_1"),df.columns.get_loc("amb3_1")+1)
r4_1   = range(df.columns.get_loc("attr4_1"),df.columns.get_loc("shar4_1")+1)
r5_1   = range(df.columns.get_loc("attr5_1"),df.columns.get_loc("amb5_1")+1)
r_s    = range(df.columns.get_loc("attr"),df.columns.get_loc("like")+1)
r_o    = range(df.columns.get_loc("attr_o"),df.columns.get_loc("like_o")+1)
r_pf_o = range(df.columns.get_loc("pf_o_att"),df.columns.get_loc("pf_o_sha")+1)
# Generate index objects from column index values.
a1_1   = df.columns[r1_1]
a2_1   = df.columns[r2_1]
a3_1   = df.columns[r3_1]
a4_1   = df.columns[r4_1]
a5_1   = df.columns[r5_1]
a_s    = df.columns[r_s] 
a_o    = df.columns[r_o]
a_pf_o = df.columns[r_pf_o]



# List to contain indices of fully NaN rows in key attribute columns.
empty = [] 
# Fill list with indices of empty rows from attributes 1_1
for index, row in df[a1_1].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill list with indices of empty rows from attributes 2_1
for index, row in df[a2_1].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill list with indices of empty rows from attributes 3_1
for index, row in df[a3_1].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        

# Fill list with indices from attributes 4_1 for waves 6 to 21
for index, row in df[df['wave'] > 5][a4_1].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill list with indices from attributes 5_1 for waves 6 to 21
for index, row in df[df['wave'] > 9][a5_1].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)

        
# Fill list with indices from attributes (scorecard answers)
for index, row in df[a_s].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill list with indices from attributes (partner's scorecard answers)
for index, row in df[a_o].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill list with indices from attributes (partner's answers to a1_1 attributes)
for index, row in df[a_pf_o].iterrows():
    if pd.isna(row).all() == True:
        empty.append(index)
        
# Fill in remaining empty cells in df[a_pf_o]. This is done because this group
# of ratings is done in point allocation. Empty cells are 0's.
df[a_pf_o].fillna(0)

# Make list empty contain unique elements.
df1 = df.copy().drop(list(set(empty)))

# At this point, unneeded columns and completely empty rows for particular 
# attribute groups have been removed from the base dataframe.



###############################################################################
# Missing Data Imputation
###############################################################################

# Generate subsets based on wave.
# Wave 6 onwards
w6on =      df1[df1['wave'] > 5]
# Waves 6 to 9
w6to9 =     df1[(df1['wave'] > 5) & (df1['wave'] < 10)]
# Waves 10 onwards
w10on =     df1[df1['wave'] > 9]
# Waves 1 to 5 and 10 to 21
wnot6to9 =  df1[(df1['wave'] < 6) | (df1['wave'] > 9)]

# Check if any NaNs exist in subset and impute them if there are.
# Imputation for attribute column group a1_1 for waves 1 to 5 and 10 to 21
if pd.isnull(wnot6to9[a1_1]).any().any():
    df1.loc[wnot6to9.index,a1_1] = imputeKNN(wnot6to9[a1_1],30)
# Imputation for attrivute column group a1_1 for waves 6 to 9
if pd.isnull(w6to9[a1_1]).any().any():    
    df1.loc[w6to9.index,a1_1] = imputeKNN(w6to9[a1_1],30)
# Imputation for attribute column group a2_1 for waves 1 to 5 and 10 to 21
if pd.isnull(wnot6to9[a2_1]).any().any():    
    df1.loc[wnot6to9.index,a2_1] =   imputeKNN(wnot6to9[a2_1],30)
# Imputation for attrivute column group a2_1 for waves 6 to 9
if pd.isnull(w6to9[a2_1]).any().any():    
    df1.loc[w6to9.index,a2_1] =  imputeKNN(w6to9[a2_1],30)
# Imputation for attribute column group a3_1
if pd.isnull(df1[a3_1]).any().any():    
    df1.loc[:,a3_1] = imputeKNN(df1[a3_1],30)
# Imputation for attribute column group a4_1 for waves 6 to 9
if pd.isnull(w6to9[a4_1]).any().any():    
    df1.loc[w6to9.index,a4_1] =  imputeKNN(w6to9[a4_1],30)
# Imputation for attribute column group a4_1 for waves 10 to 21
if pd.isnull(w6on[a4_1]).any().any():    
    df1.loc[w6on.index,a4_1] = imputeKNN(w6on[a4_1],30)
# Imputation for attribute column group a5_1 for waves 6 to 21
if pd.isnull(w10on[a5_1]).any().any():    
    df1.loc[w10on.index,a5_1] = imputeKNN(w10on[a5_1],30)
# Imputation for attribute column group a_s
if pd.isnull(df1[a_s]).any().any():    
    df1.loc[:,a_s] = imputeKNN(df1[a_s],30)
# Imputation for attribute column group a_o
if pd.isnull(df1[a_o]).any().any():    
    df1.loc[:,a_o] = imputeKNN(df1[a_o],30)
# Imputation for attribute column group a_pf_o
if pd.isnull(df1[a_pf_o]).any().any():    
    df1.loc[:,a_pf_o] = imputeKNN(df1[a_pf_o],30)

# Verify if all NaNs were imputed.
print pd.isnull(df1[a1_1]).any().any()
print pd.isnull(df1[a2_1]).any().any()
print pd.isnull(df1[a3_1]).any().any()
print pd.isnull(df1[df1['wave'] > 5][a4_1]).any().any()
print pd.isnull(df1[df1['wave'] > 5][a5_1]).any().any()
print pd.isnull(df1[a_s]).any().any()
print pd.isnull(df1[a_o]).any().any()
print pd.isnull(df1[a_pf_o]).any().any()



###############################################################################
# Data Analysis
###############################################################################
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import scale

# Question 1: Can I determine if a match will occur with ratings?
# Question Type: Binary classification
# Answer Method: Perceptron, Logistic Regression
# Features to test:     - pf_o_, _o, a1_1, a_s
#                       -
#                       -

# Question 2: 


# %% Generate Perceptron Model
X = scale(df1.loc[:,df.columns[r1_1 + r_s + r_o + r_pf_o]])
y = df1.loc[:,"match"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = Perceptron(max_iter = 1000)
clf.fit(X_train,y_train)
clf.score(X_test, y_test)








# %% Logistic Regression Model
from sklearn.linear_model import LogisticRegression
X = scale(df1.loc[:,df.columns[r1_1 + r_s + r_o + r_pf_o]])
y = df1.loc[:,"match"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)




# %% SVM Model
from sklearn import svm
X = scale(df1.loc[:,df.columns[r1_1 + r_s + r_o + r_pf_o]])
y = df1.loc[:,"match"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)




# %% Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
X = scale(df1.loc[:,df.columns[r1_1 + r_s + r_o + r_pf_o]])
y = df1.loc[:,"match"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)






# %% Generate Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Some of the points have repeated features. 
# Append new feature counting number of matches per subject
# Get individual iid numbers
idlist = [int (v) for v in list(set(df1.loc[:,'iid']))]
# Create empty list to store number of matches
nummatch = []
 # Create empty list to store first
lrlist = []
#
lrcol = [r1_1 + r2_1 + r3_1 + r4_1 + r5_1]
# loop through all of the iid numbers in dataframe df1
for i in idlist:
    # Get the subset of data for the i'th subject
    df1sub = df1.loc[df1['iid'] == i]
    # Calculate the number of matches from i'th subject subset
    msum = df1sub['match'].sum()
    # Append match sum to list nummatch
    nummatch.append(msum)
    # Get first row of subset as a list.
    lrlist.append(df1sub.values.tolist()[0])
    
# Create new subset dataframe containing nonrepeated rows of subject survey 
# answers and number of matches.
lrdf = pd.DataFrame(data=lrlist, columns = df1.columns)
# Append list containing number of subject matches to subset dataframe.
lrdf['nummatch'] = nummatch

# Linear Regression Model 1: Number of matches based on how subjects rate 
# themselves
X = lrdf.loc[:,a3_1]
y = lrdf.loc[:,'nummatch']
# Standardize input dataset.
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#reg = LinearRegression().fit(X_train,y_train)
#reg.score(X_test,y_test)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_in = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_in,y,test_size=0.2)
reg = LinearRegression().fit(X_train,y_train)
reg.score(X_test,y_test)