# -*- coding: utf-8 -*-
"""
Created on Sat May 28 10:37:41 2016

@author: Chan Family
"""


import pandas as pd
import numpy as np
import random
import distance
from stop_words import get_stop_words

random.seed(2016)

######################
## Global variables ##
######################
path = "../input/"
#!!! NEED TO EXPAND ON THESE LIST OF FEATURES (Augmenting image data?) !!!!!!#
listOfFeatures = ['itemID', 'categoryID', 'price', 'locationID', \
                    'metroID', 'lat', 'lon']
         
stop_words = get_stop_words('russian')

###############################
## reading in training files ##
###############################

def input_testing():
    # Defining format to read in pairs data and info data
    pairsType = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }

    infoType = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }
    
    print("Loading ItemPairs_train.csv")
    # pairs of items, and boolean if they are the same
    pairs = pd.read_csv(path+"ItemPairs_test.csv", dtype=pairsType)
    
    print("Loading ItemInfo_train.csv")
    # information about the items (matched on itemID)
    itemInfo = pd.read_csv(path+"ItemInfo_test.csv", dtype=infoType)
    
    # filling NA with Boolean True
    itemInfo.fillna(-1, inplace=True)
    
    # location of item (matched on locationID)
    location = pd.read_csv(path+"Location.csv")
    # category of item (matched on categoryID)
    category = pd.read_csv(path+"Category.csv")
    
    ################################
    ## Data massaging starts here ##
    ################################
    
    # padding of location and category information to overall item information
    itemInfo = pd.merge(itemInfo, category, how='left', \
                        on='categoryID', left_index=True)
    itemInfo = pd.merge(itemInfo, location, how='left',  \
                        on='locationID', left_index=True)


    # adding text features by removing stop words                        
    print('Add text features')
    itemInfo['title_corpus'] = [word for word in itemInfo['title'].str.lower().split() if word not in stop_words]
    itemInfo['description_corpus'] = [word for word in itemInfo['description'].str.lower().split() if word not in stop_words]
    itemInfo['attrsJSON_corpus'] = [word for word in itemInfo['attrsJSON'].str.lower().split() if word not in stop_words]
 
    
    # creation of info for item 1 (to match with itemID_1 in the training set)
    item1 = itemInfo[listOfFeatures] # See global variables for this definition
    
    # renaming all of them to assume role of item 1 in the pairs    
    item1 = itemInfo.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            # introducing parentCategoryID to possibly keep track of itemID?
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1',
            'title_corpus' : 'title_corpus_1',
            'description_corpus': 'description_corpus_1',
            'attrsJSON_corpus': 'attrsJSON_corpus_1'    
        }
    )
        
    # creation of info for item 2 (to match with itemID_2 in the training set)
    item2 = itemInfo[listOfFeatures]
    
    # renaming all of them to assume role of item 2 in the pairs    
    item2 = itemInfo.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2',
            'title_corpus' : 'title_corpus_2',
            'description_corpus': 'description_corpus_2',
            'attrsJSON_corpus': 'attrsJSON_corpus_2'
        }
    )
        
                        
    # padding of overall information to the items in the training set
    pairs = pd.merge(pairs, item1, how='left', on='itemID_1', left_index=True)
    pairs = pd.merge(pairs, item2, how='left', on='itemID_2', left_index=True)
    
    #renaming it for reasons unknown
    test = pairs
    
    ######################
    ## Training of data ##
    #####################
    
    # Creates an arrays to represent the result of comparing item 1 and item 2
    # In this process parentCategory is not used for comparison.
    # Hence my conclusion that its only used to keep track of items
    test['price_same'] = np.equal(test['price_1'], test['price_2']).astype(np.int32)
    test['locationID_same'] = np.equal(test['locationID_1'], test['locationID_2']).astype(np.int32)
    test['categoryID_same'] = np.equal(test['categoryID_1'], test['categoryID_2']).astype(np.int32)
    test['regionID_same'] = np.equal(test['regionID_1'], test['regionID_2']).astype(np.int32)
    test['metroID_same'] = np.equal(test['metroID_1'], test['metroID_2']).astype(np.int32)
    test['lat_same'] = np.equal(test['lat_1'], test['lat_2']).astype(np.int32)
    test['lon_same'] = np.equal(test['lon_1'], test['lon_2']).astype(np.int32)
    
    # comparing string via normalized levenshtein distance
    # threhold for 1 is set at distance 0.8
    test['title_distance'] = 1 if distance.nlevenshtein(test['title_corpus_1'],test['title_corpus_2']) \
                                > 0.8 else 0
    test['description_distance'] = 1 if distance.nlevenshtein(test['description_corpus_1'],test['description_corpus_2']) \
                                > 0.8 else 0
    test['attrsJSON_distance'] = 1 if distance.nlevenshtein(test['attrsJSON_corpus_1'],test['attrsJSON_corpus_2']) \
                                > 0.8 else 0
    
    # returning the training data
    return test

#########################
## feature engineering ##
#########################
#!!!! To expand on this for possible usage of image
