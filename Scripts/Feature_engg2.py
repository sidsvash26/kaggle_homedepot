# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:17:24 2016

@author: sidvash
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#Item selector
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


#Import last saved dataframe
df_all = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_feat.pkl') 

# TFIDF Features
tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
tsvd = TruncatedSVD(n_components=20, random_state=0)  #dimension reduction

feat = FeatureUnion(transformer_list=[('query', Pipeline([('selector', ItemSelector(key='search_term')), ('tfidf', tfidf), ('tsvd', tsvd)])), ('title', Pipeline([('selector', ItemSelector(key='product_title')), ('tfidf', tfidf), ('tsvd', tsvd)])), ('desc', Pipeline([('selector', ItemSelector(key='product_description')),('tfidf', tfidf), ('tsvd', tsvd)])), ('brand', Pipeline([('selector', ItemSelector(key='brand')), ('tfidf', tfidf), ('tsvd', tsvd)]))])

tfidf_feat = feat.fit_transform(df_all)
tfidf_df = pd.DataFrame(tfidf_feat)

#concat features to main dataframe
df_all = pd.concat([df_all, tfidf_df], axis=1)

#Saving dataframe
df_all.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_feat2.pkl') 



