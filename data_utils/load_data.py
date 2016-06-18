import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
import time
from Levenshtein import distance


class Featurizer():

    def _add_item_features(self, items):
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

    def _add_pairs_features(self, pairs, split):
        """
        Do feature engineering here by adding features to each item pairs, i.e.
        features common to both items in the pair
        :param pairs: itempairs df
        :params split: either "train" or "test" to indicate split
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
        pairs['parentCategoryID_same'] = np.equal(
            pairs['parentCategoryID_1'],
            pairs['parentCategoryID_2']).astype(
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

        # Add l2 distance of lon/lat
        print("Adding l2 distances")
        pairs['geo_l2'] = pairs.apply(lambda x: np.linalg.norm(
            [x.lon_1-x.lon_2, x.lat_1-x.lat_2]), axis=1)

        # Add precomputed edit distances (using fuzzywuzzy)
        print("Adding precomputed edit distances")
        try:
            edit = pd.read_csv('itempairs_%s_edit_distances.csv' % split)
            pairs = pd.merge(pairs, edit, how='left',
                             on=['itemID_1', 'itemID_2'])
        except:
            raise Exception("Invalid split specified or file does not exist")

        # Add consine sim based on doc2vec vectors
        print("Adding doc2vec cos sim")
        pairs['d2v_sim'] = pairs[
            ['itemID_1', 'itemID_2']].apply(
            lambda x, self=self: self._desc_d2v.docvecs.similarity(x[0], x
                                                                   [1]), axis=1)

        # Compute JSON edit distances
        # TODO: maybe count individual value edit distances, count number of
        # different keys
        print("Adding json edit distance")
        pairs['json_edit'] = pairs[['attrsJSON_1', 'attrsJSON_2']].apply(
            lambda x: distance(str(x[0]), str(x[1])), axis=1)

        pairs.drop(['title_1', 'title_2', 'description_1', 'description_2',
                    'attrsJSON_1', 'attrsJSON_2'], axis=1, inplace=True)
        return pairs

    def _merge_itempairs(
            self,
            iteminfo_train,
            iteminfo_test,
            train_egs=None,
            test_egs=None):
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
        itempairs_train = pd.merge(
            itempairs_train,
            iteminfo_train_1,
            how='left',
            on='itemID_1',
            left_index=True)
        iteminfo_train_2 = iteminfo_train.copy(deep=True)
        iteminfo_train_2.rename(columns=lambda x: x+'_2', inplace=True)
        itempairs_train = pd.merge(
            itempairs_train,
            iteminfo_train_2,
            how='left',
            on='itemID_2',
            left_index=True)
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

        # Randomly sample subset of ItemPairs
        if train_egs:
            itempairs_train = itempairs_train.sample(n=train_egs)
        if test_egs:
            itempairs_test = itempairs_test.sample(n=test_egs)
        return itempairs_train, itempairs_test

    def _get_features(self, train, test):
        trainval = list(train.columns.values)
        testval = list(test.columns.values)
        output = list(set(trainval) & set(testval))
        output.remove('itemID_1')
        output.remove('itemID_2')
        return output

    def __init__(self):
        self._desc_d2v = Doc2Vec.load('description.d2v')


def load_data(train_egs=None, test_egs=None):
    """
    Load ItemInfo, do feature engineering on ItemInfo, then merge into ItemPairs,
    do feature engineering on ItemPairs, then return train/test examples and
    features used
    :param train_egs: Randomly sample train_egs ItemPair examples for training
    :param test_egs: Randomly sample test_egs ItemPair examples for testing
    :return: train df, test df and list of features
    """
    featurizer = Featurizer()

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
    start_time = time.time()
    iteminfo_train = featurizer._add_item_features(iteminfo_train)
    iteminfo_test = featurizer._add_item_features(iteminfo_test)
    print(
        'Feature eng ItemInfo time: {} minutes'.format(
            round(
                (time.time() - start_time)/60,
                2)))

    # Merge iteminfo onto itempairs
    itempairs_train, itempairs_test = featurizer._merge_itempairs(
        iteminfo_train, iteminfo_test, train_egs, test_egs)

    # Feature engineering on ItemPairs, i.e. pair of items
    start_time = time.time()
    itempairs_train = featurizer._add_pairs_features(itempairs_train, "train")
    itempairs_test = featurizer._add_pairs_features(itempairs_test, "test")
    print(
        'Feature eng ItemPairs time: {} minutes'.format(
            round(
                (time.time() - start_time)/60,
                2)))

    # Extract features
    features = featurizer._get_features(itempairs_train, itempairs_test)
    return itempairs_train, itempairs_test, features
