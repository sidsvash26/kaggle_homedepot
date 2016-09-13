# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:05:52 2016

@author: sidvash
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import train_test_split

#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#define RMSE
def f_mse(ground_truth, predictions):
    f_mse = mean_squared_error(ground_truth, predictions)**0.5
    return f_mse
RMSE = make_scorer(f_mse, greater_is_better=False)


#Import
df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
num_test = df_test.shape[0]

df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_feat2.pkl') 

#Seperate origianl training and testing dataset
df_train = df1.iloc[:num_train]
df_test = df1.iloc[num_train:]
df_test = df_test.drop("relevance", axis=1)

#Missing value imputation
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

#Create X and Y
X = df_train.drop(['id', 'product_title', 'product_uid', 'relevance', 'search_term', 'product_description', 'brand', 'product_info'], axis=1)
y = df_train['relevance']
X_test = df_test.drop(['id', 'product_title', 'product_uid', 'search_term', 'product_description', 'brand', 'product_info'], axis=1)

#Grid Search CV
xgb = xgb.XGBRegressor()

param_grid = [{'n_estimators': [500] , 'max_depth':[25] } ]

clf = GridSearchCV(estimator = xgb, param_grid = param_grid, cv=5, scoring=RMSE)

clf.fit(X,y)

clf.predict(X_test)
#best score --->   -0.48685505725941314
output_xgb = pd.DataFrame({"id": id_test, "relevance": clf_pred})
output_xgb.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub8_finalModel_xgb.csv',index=False)

#************************     RFR Model   *************************

rfr = RandomForestRegressor(n_jobs=-1, random_state=0)
clf2 = GridSearchCV(estimator = rfr, param_grid = param_grid, cv=5, scoring=RMSE)
clf2.best_score #0.46281826589507041

clf2_pred = clf2.predict(X_test)
id_test = df_test["id"]

#Export
output1 = pd.DataFrame({"id": id_test, "relevance": clf2_pred})

output1.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub8_finalModel_rfr.csv',index=False)

