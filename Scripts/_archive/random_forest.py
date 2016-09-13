# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:03:03 2016

@author: sidvash
"""
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import numpy as np
import pandas as pd
#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
num_test = df_test.shape[0]


df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')

# drop certain features for modelling
df1.drop(['search_term','product_title','product_description','product_info', 'brand', 'product_uid'],axis=1, inplace=True)




#***********************************    Modelling    *********************************************

df_train = df1.iloc[:num_train]
df_test = df1.iloc[num_train:]
id_test_orig = df_test['id']
df_test_orig = id_test_orig.to_frame()


#drop where no search queries are present
df_train = df_train[df_train.len_query != 0]
df_test = df_test[df_test.len_query != 0]
id_test_temp = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Find accuracy 
ytrain_pred = clf.predict(X_train)

RMSE = np.sqrt(sum((y_train - ytrain_pred)*(y_train - ytrain_pred))/len(y_train))
print(RMSE)

#prediction
df_test_temp = pd.DataFrame({"id": id_test_temp, "relevance": y_pred})
df_test_orig = pd.merge(df_test_orig, df_test_temp, how='left', on='id') 
df_test_orig['relevance'].fillna(1, inplace=True)

df_test_orig.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub4.csv',index=False)