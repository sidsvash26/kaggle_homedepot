# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:38:44 2016

@author: sidvash
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
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
        s = re.sub(r"([a-zA-Z])\/([a-zA-Z])", r"\1 \2", s)  #sep '/' b/w letters
        
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
 

 # ***************   Description and brand pre-process ************

#Pre-process description and attributes seperately before merging
df_prod["product_description"] = df_prod["product_description"].map(lambda x: string_replace(x))
df_prod["product_description"] = df_prod["product_description"].map(lambda x: remove_sw(x))
df_prod["product_description"] = df_prod["product_description"].map(lambda x: stem(x))
df_prod['len_desc'] = df_prod['product_description'].map(lambda x: len_string(x))


df_brand["brand"] = df_brand["brand"].map(lambda x: string_replace(x))
df_brand["brand"] = df_brand["brand"].map(lambda x: remove_sw(x))
df_brand["brand"] = df_brand["brand"].map(lambda x: stem(x))

#Merge description, attributes
df_all = pd.merge(df_all, df_prod, how='left', on='product_uid') 
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all.brand.fillna(" ", inplace=True)

 # ***************   Main data frame pre-processing ************
df_all["search_term"] = df_all["search_term"].map(lambda x: string_replace(x))
df_all["search_term"] = df_all["search_term"].map(lambda x: remove_sw(x))
df_all["search_term"] = df_all["search_term"].map(lambda x: stem(x))

df_all["product_title"] = df_all["product_title"].map(lambda x: string_replace(x))
df_all["product_title"] = df_all["product_title"].map(lambda x: remove_sw(x))
df_all["product_title"] = df_all["product_title"].map(lambda x: stem(x))

#************************   Intersecting Terms (common terms) ***************************

df_all["product_info"] = df_all["search_term"] + "\t" + df_all["product_title"] + "\t" +  df_all["product_description"] + "\t" +df_all["brand"]  
           
# COMMON UNIGRAMS            
df_all['word_in_title'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[1]))    
df_all['word_in_description'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[2]))  
df_all['word_in_brand'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[3]))    
            
df_all.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df1.pkl')        
df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df1.pkl')

#number of unigrams in a string
df1["len_query"] = df1["search_term"].map(lambda x: len_string(x))
df1["len_title"] = df1["product_title"].map(lambda x: len_string(x) )
df1["len_description"] = df1["product_description"].map(lambda x: len_string(x))

# COMMON Bi-GRams and TRI-GRAMS
df1['bigram_in_title'] = df1['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[1])) 
df1['trigram_in_title'] = df1['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[1]))

df1['bigram_in_desc'] = df1['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[2]))  
df1['trigram_in_desc'] = df1['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[2]))

 #ratio of common words in title to that of length 
df1['ratio_title'] = df1['word_in_title']/df1['len_query']
df1['ratio_desc'] = df1['word_in_description']/df1['len_query']


#Save dataframe
df1.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl') 

df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')













        
    