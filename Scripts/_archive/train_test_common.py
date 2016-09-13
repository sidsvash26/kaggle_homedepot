# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:46:58 2016

@author: sidvash
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

#Only for spyder 


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


#************************** Pre-processing Functions********************************

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# Basic Pre-processing 

  #function for basic replacements:
def string_replace(s):
    if isinstance(s, str):
       #Seperators        
        s=s.replace("-", " ")
        s = re.sub(r"([a-zA-Z])\.([a-zA-Z)])", r"\1 \2", s) #sep '.' b/w letters
        s=re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", s) #sep. alpha anumeric
        s=re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([a-zA-Z])/([a-zA-Z])", r"\1 \2", s)  #sep '/' b/w letters
        
       #Substitutions
        s=s.replace(" ac ", " air condition ")
        s=s.replace("airconditioner", "air condition")
        s=s.replace("toliet", "toilet")
        s=s.replace("tiolet", "toilet")
        s=s.replace("sprkinler", "sprinkler")
        s=s.replace("bathro", "bath room")
        s=s.replace("bathroom", "bath room")
        s=s.replace("vlve", "valve")
        return s
    else:
        return "null"

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

#************************** Counting and Intersecting Features (n-gram) ************************    
#No. of words in a string -length
def len_string(string1):
    words_s1 = word_tokenize(string1.lower())
    return len(words_s1)


#No. of common words in two strings - UNIGRAM
def common_unigram(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    common_words = [x for x in words_s1 if x in words_s2]
    return len(common_words)

#No of common BI-GRAM
def common_bigrams(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    bigram_s1 = list(zip(words_s1, words_s1[1:]))
    bigram_s2 = list(zip(words_s2, words_s2[1:]))
    common_bigrams = [x for x in bigram_s1 if x in bigram_s2]
    return len(common_bigrams)
 
#No of common TRI-GRAMS
def common_trigrams(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    trigram_s1 = list(zip(words_s1, words_s1[1:], words_s1[2:]))
    trigram_s2 = list(zip(words_s2, words_s2[1:], words_s2[2:]))
    common_trigrams = [x for x in trigram_s1 if x in trigram_s2]
    return len(common_trigrams)
    
    
    
df_train["search_term"] = df_train["search_term"].map(lambda x: string_replace(x))
df_train["search_term"] = df_train["search_term"].map(lambda x: remove_sw(x))
df_train["search_term"] = df_train["search_term"].map(lambda x: stem(x))


df_test["search_term"] = df_test["search_term"].map(lambda x: string_replace(x))
df_test["search_term"] = df_test["search_term"].map(lambda x: remove_sw(x))
df_test["search_term"] = df_test["search_term"].map(lambda x: stem(x))    

df_all = pd.merge(df_train, df_test, how='inner', on='product_uid')
df_all["prod_info"] = df_all["search_term_x"] + "\t" + df_all["search_term_y"] 

df_all['comm_unigrams'] = df_all['prod_info'].map(lambda x: common_unigram(x.split('\t')[1],x.split('\t')[0]))      
df_all['comm_bigrams'] = df_all['prod_info'].map(lambda x: common_bigrams(x.split('\t')[1],x.split('\t')[0]))          
df_all['comm_trigrams'] = df_all['prod_info'].map(lambda x: common_trigrams(x.split('\t')[1],x.split('\t')[0]))                       
df_all['len_query_y'] = df_all['search_term_y'].map(lambda x:len_string(x))
df_all['ratio_unigram'] = df_all['comm_unigrams']/df_all['len_query_y']  
df_all['len_query_x'] = df_all['search_term_x'].map(lambda x:len_string(x)) 


df_temp = df_all[(df_all.ratio_unigram ==1) & (df_all.len_query_x == df_all.len_query_y) ]
df_temp =  df_temp[['id_y', 'relevance']]
df_temp = df_temp.drop_duplicates('id_y')


# Reading previous submission 
df_sub = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub5_gbm_newdf.csv', encoding="ISO-8859-1")

df_sub1 = pd.merge(df_sub, df_temp, how='left', left_on='id', right_on='id_y')

#replacing relevance values taken from df_temp 
df_sub1.loc[df_sub1.id == df_sub1.id_y, 'relevance_x'] = df_sub1.loc[df_sub1.id == df_sub1.id_y, 'relevance_y']
df_sub1.drop(['id_y', 'relevance_y'],axis=1, inplace=True)
df_sub1.columns = ['id', 'relevance']

df_sub1.to_csv('/home/sidvash/kaggle_2016/home_depot/submissions/sub6_gbm_newdf_replace.csv', index=False)


 