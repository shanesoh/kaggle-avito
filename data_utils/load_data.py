import pandas as pd
import numpy as np
from Levenshtein import distance

def _add_item_features(items):
    """
    Do feature engineering here by adding features to each individual item, i.e.
    features specific to each item
    :param items: iteminfo df
    """
    # Add lens of texts
    print("Adding lens of text")
    items['len_title'] = items['title'].str.len()
    items['len_description'] = items['description'].str.len()
    items['len_attrsJSON'] = items['attrsJSON'].str.len()

    items.drop(['images_array'], axis=1, inplace=True)
    return items


def _add_pairs_features(pairs):
    """
    Do feature engineering here by adding features to each item pairs, i.e.
    features common to both items in the pair
    :param pairs: itempairs df
    """
    # Add binary features indicating if attributes are equal
    print("Adding binary features")
    pairs['price_same'] = np.equal(
        pairs['price_1'],
        pairs['price_2']).astype(
        np.int32)
    pairs['locationID_same'] = np.equal(
        pairs['locationID_1'],
        pairs['locationID_2']).astype(
        np.int32)
    pairs['categoryID_same'] = np.equal(
        pairs['categoryID_1'],
        pairs['categoryID_2']).astype(
        np.int32)
    pairs['regionID_same'] = np.equal(
        pairs['regionID_1'],
        pairs['regionID_2']).astype(
        np.int32)
    pairs['metroID_same'] = np.equal(
        pairs['metroID_1'],
        pairs['metroID_2']).astype(
        np.int32)
    pairs['lat_same'] = np.equal(
        pairs['lat_1'],
        pairs['lat_2']).astype(
        np.int32)
    pairs['lon_same'] = np.equal(
        pairs['lon_1'],
        pairs['lon_2']).astype(
        np.int32)

    # Normalized edit distance of texts
    print("Adding edit distance")
    pairs['title_dist'] = pairs[['title_1', 'title_2']].astype(str).apply(
        lambda x: distance(x[0], x[1]) / float(len(x[0]) + len(x[1])),
        axis=1)
    pairs['description_dist'] = pairs[['description_1', 'description_2']].astype(str).apply(
        lambda x: distance(x[0], x[1]) / float(len(x[0]) + len(x[1])),
        axis=1)
    pairs['attrsJSON_dist'] = pairs[['attrsJSON_1', 'attrsJSON_2']].astype(str).apply(
        lambda x: distance(x[0], x[1]) / float(len(x[0]) + len(x[1])),
        axis=1)
    pairs.drop(['title_1', 'title_2', 'description_1', 'description_2',
                'attrsJSON_1', 'attrsJSON_2'], axis=1, inplace=True)

    return pairs


def load_data():
    """
    Load ItemInfo, do feature engineering on ItemInfo, then merge into ItemPairs,
    do feature engineering on ItemPairs, then return train/test examples and
    features used
    :return: train df, test df and list of features
    """
    types = {
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
    print("Load ItemInfo_train.csv")
    iteminfo_train = pd.read_csv("./data/ItemInfo_train.csv", dtype=types)
    iteminfo_train.fillna(-1, inplace=True)

    print("Load ItemInfo_test.csv")
    iteminfo_test = pd.read_csv("./data/ItemInfo_test.csv", dtype=types)
    iteminfo_test.fillna(-1, inplace=True)

    # Add in location and category data
    location = pd.read_csv("./data/Location.csv")
    category = pd.read_csv("./data/Category.csv")
    iteminfo_train = pd.merge(
        iteminfo_train,
        category,
        how='left',
        on='categoryID',
        left_index=True)
    iteminfo_train = pd.merge(
        iteminfo_train,
        location,
        how='left',
        on='locationID',
        left_index=True)
    iteminfo_test = pd.merge(
        iteminfo_test,
        category,
        how='left',
        on='categoryID',
        left_index=True)
    iteminfo_test = pd.merge(
        iteminfo_test,
        location,
        how='left',
        on='locationID',
        left_index=True)

    # Feature engineering on ItemInfo, i.e. individual items
    iteminfo_train = _add_item_features(iteminfo_train)
    iteminfo_test = _add_item_features(iteminfo_test)

    # Merge iteminfo onto itempairs
    itempairs_train, itempairs_test = _merge_itempairs(
        iteminfo_train, iteminfo_test)

    # Feature engineering on ItemPairs, i.e. pair of items
    itempairs_train = _add_pairs_features(itempairs_train)
    itempairs_test = _add_pairs_features(itempairs_test)

    # Extract features
    features = _get_features(itempairs_train, itempairs_test)
    return itempairs_train, itempairs_test, features


def _merge_itempairs(iteminfo_train, iteminfo_test):
    types = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }
    print("Load ItemPairs_train.csv")
    itempairs_train = pd.read_csv("./data/ItemPairs_train.csv", dtype=types)
    print("Load ItemPairs_test.csv")
    itempairs_test = pd.read_csv("./data/ItemPairs_test.csv", dtype=types)

    # Merge itempairs for train set
    iteminfo_train_1 = iteminfo_train.copy(deep=True)
    iteminfo_train_1.rename(columns=lambda x: x+'_1', inplace=True)
    itempairs_train = pd.merge(itempairs_train, iteminfo_train_1, how='left',
                               on='itemID_1', left_index=True)
    iteminfo_train_2 = iteminfo_train.copy(deep=True)
    iteminfo_train_2.rename(columns=lambda x: x+'_2', inplace=True)
    itempairs_train = pd.merge(itempairs_train, iteminfo_train_2, how='left',
                               on='itemID_2', left_index=True)
    # 'generationMethod' exists only in train, so drop
    itempairs_train.drop('generationMethod', axis=1, inplace=True)

    # Merge itempairs for test set
    iteminfo_test_1 = iteminfo_test.copy(deep=True)
    iteminfo_test_1.rename(columns=lambda x: x+'_1', inplace=True)
    itempairs_test = pd.merge(itempairs_test, iteminfo_test_1, how='left',
                              on='itemID_1', left_index=True)
    iteminfo_test_2 = iteminfo_test.copy(deep=True)
    iteminfo_test_2.rename(columns=lambda x: x+'_2', inplace=True)
    itempairs_test = pd.merge(itempairs_test, iteminfo_test_2, how='left',
                              on='itemID_2', left_index=True)

    return itempairs_train, itempairs_test


def _get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = list(set(trainval) & set(testval))
    output.remove('itemID_1')
    output.remove('itemID_2')
    return output
