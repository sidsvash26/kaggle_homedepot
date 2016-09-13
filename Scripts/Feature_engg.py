# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 22:59:46 2016

@author: sidvash
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import re

#Only for spyder -removes run time warning messages
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

df_all = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_preProcess.pkl')

#No. of words in a string -length
def len_string(string1):
    words_s1 = word_tokenize(string1.lower())
    return len(words_s1)
def no_of_bigrams(string1):
    words_s1= word_tokenize(string1.lower())
    bigram_s1= list(zip(words_s1, words_s1[1:]))
    return len(bigram_s1)
def no_of_trigrams(string1):
    words_s1 = word_tokenize(string1.lower())
    trigram_s1 = list(zip(words_s1, words_s1[1:], words_s1[2:]))
    return len(trigram_s1)
    
        
    

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

#Variable for common features
df_all["product_info"] = df_all["search_term"] + "\t" + df_all["product_title"] + "\t" +  df_all["product_description"] + "\t" +df_all["brand"] 

#****************   Length Features ******************

#number of unigrams in a string
df_all["len_query"] = df_all["search_term"].map(lambda x: len_string(x))
df_all["len_title"] = df_all["product_title"].map(lambda x: len_string(x) )
df_all["len_description"] = df_all["product_description"].map(lambda x: len_string(x))
df_all["len_brand"] = df_all["brand"].map(lambda x: len_string(x))

#number of bigrams in a string 
df_all["len_2_query"] = df_all["search_term"].map(lambda x: no_of_bigrams(x))
df_all["len_2_title"] = df_all["product_title"].map(lambda x: no_of_bigrams(x) )
df_all["len_2_description"] = df_all["product_description"].map(lambda x: no_of_bigrams(x))
df_all["len_2_brand"] = df_all["brand"].map(lambda x: no_of_bigrams(x))

#number of trigrams in a string
df_all["len_3_query"] = df_all["search_term"].map(lambda x: no_of_trigrams(x))
df_all["len_3_title"] = df_all["product_title"].map(lambda x: no_of_trigrams(x) )
df_all["len_3_description"] = df_all["product_description"].map(lambda x: no_of_trigrams(x))
df_all["len_3_brand"] = df_all["brand"].map(lambda x: no_of_trigrams(x))

#*****************     Intersecting Features  *********************

#Unigrams common b/w search term and other features
df_all['unig_in_title'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[1]))    
df_all['unig_in_desc'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[2]))  
df_all['unig_in_brand'] = df_all['product_info'].map(lambda x: common_unigram(x.split('\t')[0],x.split('\t')[3]))    

#Common bi-grams
df_all['bigram_in_title'] = df_all['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[1])) 
df_all['bigram_in_desc'] = df_all['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[2]))  
df_all['bigram_in_brand'] = df_all['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[3]))  


#Common tri-grams
df_all['trigram_in_title'] = df_all['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[1]))
df_all['trigram_in_desc'] = df_all['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[2]))
df_all['trigram_in_brand'] = df_all['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[3]))

#**** Intersecting Ratios ****
#unigram ratios
df_all['ratio_uni_title'] = df_all['unig_in_title']/df_all['len_query']
df_all['ratio_uni_desc'] = df_all['unig_in_desc']/df_all['len_query']
df_all['ratio_uni_brand'] = df_all['unig_in_brand']/df_all['len_query']

#bigram ratios
df_all['ratio_bi_title'] = df_all['bigram_in_title']/df_all['len_2_query']
df_all['ratio_bi_desc'] = df_all['bigram_in_desc']/df_all['len_2_query']
df_all['ratio_bi_brand'] = df_all['bigram_in_brand']/df_all['len_2_query']

#trigram ratios
df_all['ratio_tri_title'] = df_all['trigram_in_title']/df_all['len_3_query']
df_all['ratio_tri_desc'] = df_all['trigram_in_desc']/df_all['len_3_query']
df_all['ratio_tri_brand'] = df_all['trigram_in_brand']/df_all['len_3_query']

df_all.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_feat.pkl') 

