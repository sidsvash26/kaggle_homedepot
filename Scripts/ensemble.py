# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:17:36 2016

@author: sidvash
"""

#Ensemble 
import pandas as pd

#Import kaggle script 0.47385 
sub1 = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub8_finalModel_rfr.csv')

sub2 = pd.read_csv('/home/sidvash/Downloads/submission.csv')

sub1['avg_rel'] = (sub1['relevance'] + sub2['relevance'])/2
sub1.drop('relevance', axis=1, inplace=True)
sub1.columns = ['id', 'relevance']

sub1.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub9_ens_8andkaggle.csv',index=False)


