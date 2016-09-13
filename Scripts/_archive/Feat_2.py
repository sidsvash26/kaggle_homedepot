# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:26:57 2016

@author: sidvash
"""
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.metrics import *


#Only for spyder 
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


df_train = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/home/sidvash/kaggle_2016/home_depot/raw_data/test.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
num_test = df_test.shape[0]

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
    common_words = [x for x in words_s1 if x in words_s2]
    return len(common_words)

#No of common bi-grams
def common_bigrams(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    bigram_s1 = list(zip(words_s1, words_s1[1:]))
    bigram_s2 = list(zip(words_s2, words_s2[1:]))
    common_bigrams = [x for x in bigram_s1 if x in bigram_s2]
    return len(common_bigrams)
 
#No of common tri-grams   
def common_trigrams(string1, string2):
    words_s1 = word_tokenize(string1.lower())
    words_s2 = word_tokenize(string2.lower())
    trigram_s1 = list(zip(words_s1, words_s1[1:], words_s1[2:]))
    trigram_s2 = list(zip(words_s2, words_s2[1:], words_s2[2:]))
    common_trigrams = [x for x in trigram_s1 if x in trigram_s2]
    return len(common_trigrams)
 
#No. of words in a string -length
def len_string(string1):
    words_s1 = word_tokenize(string1.lower())
    return len(words_s1) 
    
# Further Pre-processing    
    
 
#******************************* Import last dataframe  *********************
    
df1 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df1.pkl')   
    
#**************************  add common n grams feature ******************************

df1['bigram_in_title'] = df1['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[1])) 
df1['trigram_in_title'] = df1['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[1]))

df1['bigram_in_desc'] = df1['product_info'].map(lambda x: common_bigrams(x.split('\t')[0],x.split('\t')[2]))  
df1['trigram_in_desc'] = df1['product_info'].map(lambda x: common_trigrams(x.split('\t')[0],x.split('\t')[2]))

#save dataframe as df2
df1.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')
   
#*********************************** Length and ratio queries *****************************
 #Load last data frame 
df2 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')  

 
df2['len_query'] = df2['search_term'].map(lambda x: len_string(x))
df2['len_title'] = df2['product_title'].map(lambda x: len_string(x)) 
df2['len_desc'] =  df2['product_description'].map(lambda x: len_string(x)) 
#df2['len_brand'] = df2['brand'].map(lambda x: len_string(x)) 

      #ratio of common words in title to that of length 
df2['ratio_title'] = df2['word_in_title']/df2['len_query']
df2['ratio_desc'] = df2['word_in_description']/df2['len_query']
#df2['ratio_brand'] = df2['word_in_brand']/df2['len_brand']
  
df2.to_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')  

  
#*****************************  Further Pre-processing  **********************
df2 = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df2.pkl')  

  #function for basic replacements:
#
'''
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
        s=s.replace("toliet", "toilet")
        s=s.replace("tiolet", "toilet")
        s=s.replace("sprkinler", "sprinkler")
        s=s.replace("bathro", "bath room")
        s=s.replace("bathroom", "bath room")
        s=s.replace("vlve", "valve")
        return s
    else:
        return "null"

'''        
  #Edit_distance between query and brand name
    
  
  
  
  
  
  
  
  
