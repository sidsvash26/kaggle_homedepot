# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:31:58 2016

@author: sidvash
"""

import re, math
from collections import Counter
import Levenshtein
import pandas as pd
from nltk.metrics.distance import jaccard_distance


WORD = re.compile(r'\w+')
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
     
def get_cosine(string1, string2):
     vec1 = text_to_vector(string1)
     vec2 =text_to_vector(string2)

     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator     

def jaccard(string1, string2):
    n=3
    set1 = set([string1[i:i+n] for i in range(0, len(string1), n)])
    set2 = set([string2[i:i+n] for i in range(0, len(string2), n)])
    return jaccard_distance(set1, set2)

        
    
    
    
df_all = pd.read_pickle('/home/sidvash/kaggle_2016/home_depot/dataframes/df_feat2.pkl') 

################# DIstance Features ####################

df_all['cosine_dist'] = df_all['product_info'].map(lambda x: get_cosine(x.split('\t')[0],x.split('\t')[1]))
