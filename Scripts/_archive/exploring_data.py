# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:43:39 2016

@author: sidvash
"""


import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

#Importing
df_prod = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/product_descriptions.csv')
df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/attributes.csv', encoding="ISO-8859-1")

df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value":"brand"})
df_brand.brand.fillna(" ", inplace=True)

num_train = df_train.shape[0]
num_test = df_test.shape[0]

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)


#************************ Pre-processing Functions********************************

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

#A function that eliminates the stopwords in a string 
def remove_sw(string):
    words = word_tokenize(string)
    filtered_sentence = []
    for word in words:
        if word not in stop_words:
            filtered_sentence.append(word)
    final_sentence = " ".join(filtered_sentence)        
    return final_sentence;
      
#A function that stems all the words in a string          
def stem(string):
    words = word_tokenize(string)
    stem_sentence = []
    for word in words:
        stem = ps.stem(word)
        stem_sentence.append(stem)
    final_sentence = " ".join(stem_sentence)
    return final_sentence
    
#No. of common words in two strings
def common_num(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    common_words = [x for x in words_s2 if x in words_s1]
    return len(common_words)
    
#No. of words in a string -length
def len_string(string1):
    words_s1 = word_tokenize(string1.lower())
    return len(words_s1)



#merge desc, attributes
df_all = pd.merge(df_all, df_prod, how='left', on='product_uid') 
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all.brand.fillna(" ", inplace=True)
