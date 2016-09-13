# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:03:03 2016

@author: sidvash
"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier


#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
num_test = df_test.shape[0]

#Import data frame
df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')
df1.drop(['search_term','product_title','product_description','product_info', 'brand'],axis=1, inplace=True)

# Modelling *********************************

df_train = df1.iloc[:num_train]
df_test = df1.iloc[num_train:]
id_test_orig = df_test['id']
df_test_orig = id_test_orig.to_frame()


#drop where no search queries are present
df_train = df_train[df_train.len_query != 0]
df_test = df_test[df_test.len_query != 0]
id_test_temp = df_test['id']

#convert relevance as a label
Y_train = df_train["relevance"].map(lambda x: str(x))

X_train = df_train.drop(['id','relevance'],axis=1)
X_test = df_test.drop(['id','relevance'],axis=1).values

#model fit
clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(X_train, Y_train)

#prediction
preds = clf.predict(X_train)
Y_train = Y_train.to_frame()
Y_train['preds'] = preds

#convert back to float
Y_train['relevance'] = Y_train["relevance"].map(lambda x: float(x))
Y_train['preds'] = Y_train["preds"].map(lambda x: float(x))

# Root mean squared error
Y_train['squared error'] = (Y_train['relevance'] - Y_train['preds'])*(Y_train['relevance'] - Y_train['preds'])
RMSE = np.sqrt(sum(Y_train['squared error'])/len(Y_train['squared error']))

