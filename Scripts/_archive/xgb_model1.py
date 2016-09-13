# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:58:52 2016

@author: sidvash
"""

import numpy as np
import pandas as pd
import xgboost as xgb

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

gbm = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
predictions = gbm.predict(X_train)

#Capping maximum prediction value to 3
for i in range(len(predictions)-1):
    if predictions[i] >=3:
        predictions[i] = 3
#Capping minimum prediction value to 1       
for i in range(len(predictions)-1):
    if predictions[i] <=1:
        predictions[i] = 1        
#Check accuracy on training data
RMSE = np.sqrt(sum((y_train - predictions)*(y_train - predictions))/len(y_train))
print(RMSE)

#prediction on Testing data
y_pred = gbm.predict(X_test)
df_test_temp = pd.DataFrame({"id": id_test_temp, "relevance": y_pred})
df_test_orig = pd.merge(df_test_orig, df_test_temp, how='left', on='id') 
df_test_orig['relevance'].fillna(1, inplace=True)


# Capping max relevance value to 3
df_test_orig.loc[df_test_orig.relevance >=3 , 'relevance'] = 3
df_test_orig.loc[df_test_orig.relevance <=1 , 'relevance'] = 1


#Exporiting file
df_test_orig.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub5_gbm.csv',index=False)



