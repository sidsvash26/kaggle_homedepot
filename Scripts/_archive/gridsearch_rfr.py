# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 19:06:35 2016

@author: sidvash
"""


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#define RMSE
def f_mse(ground_truth, predictions):
    f_mse = mean_squared_error(ground_truth, predictions)**0.5
    return f_mse
RMSE = make_scorer(f_mse, greater_is_better=False)


#Item selector
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]



df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
num_test = df_test.shape[0]

df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')

#Seperate training and test datasets
df_train = df1.iloc[:num_train]
df_test = df1.iloc[num_train:]

df_train.fillna(-1, inplace=True)


#Adding extra tfidf features
tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
#feat = tfidf.fit_transform(df_train.search_term)
#tfidf.get_feature_names()
tsvd = TruncatedSVD(n_components=10, random_state=0)

feat = FeatureUnion(transformer_list=[('query', Pipeline([('selector', ItemSelector(key='search_term')), ('tfidf', tfidf), ('tsvd', tsvd)])), ('title', Pipeline([('selector', ItemSelector(key='product_title')), ('tfidf', tfidf), ('tsvd', tsvd)])), ('desc', Pipeline([('selector', ItemSelector(key='product_description')),('tfidf', tfidf), ('tsvd', tsvd)])), ('brand', Pipeline([('selector', ItemSelector(key='brand')), ('tfidf', tfidf), ('tsvd', tsvd)]))])
extra_train_feat = feat.fit_transform(df_train)
extra_train_feat = pd.DataFrame(extra_train_feat)

df_train = pd.concat([df_train, extra_train_feat], axis=1)


#Create X and Y
X = df_train.drop(['id', 'product_title', 'product_uid', 'relevance', 'search_term', 'product_description', 'brand', 'product_info'], axis=1)
y = df_train['relevance']

#cross validation split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

param_grid = [{'n_estimators': [50,100,150,300,350,400] , 'max_depth':[6,10,15,20] } ]
rfr = RandomForestRegressor(n_jobs=-1, random_state=0)
clf = GridSearchCV(estimator = rfr, param_grid = param_grid, cv=5, scoring=RMSE)
clf.fit(X_train,y_train)
#clf.best_params_
#Out[22]: {'max_depth': 10, 'n_estimators': 400}

#clf.best_score_
#Out[23]: -0.48495465320070369

#with tfidf
#clf.best_params_
#Out[25]: {'max_depth': 20, 'n_estimators': 400}

#clf.best_score_
#Out[26]: -0.4686341639545295

#Final submission df_test:

df_test.drop('relevance', axis=1, inplace=True)
df_test.fillna(-1, inplace=True)
test_index = df_test.index.values
id_test_temp = df_test['id']


extra_test_feat = feat.fit_transform(df_test)
extra_test_df = pd.DataFrame(extra_test_feat, index=test_index)

df_test1 = pd.concat([df_test, extra_test_df], axis=1)
df_test1.drop(['id', 'product_title', 'product_uid', 'search_term', 'product_description', 'brand', 'product_info'], axis=1, inplace=True)

preds_test = clf.predict(df_test1)

df_test_temp = pd.DataFrame({"id": id_test_temp, "relevance": preds_test})
df_test_temp.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub7_tfidf_rfr.csv',index=False)
