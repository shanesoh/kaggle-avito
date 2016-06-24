"""
For offline generation of features that take a long time
"""

import numpy as np
from fuzzywuzzy import fuzz
import pandas as pd
from data_utils.load_data import Featurizer
import time
from Levenshtein import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import hamming_loss


class OfflineFeaturizer(Featurizer):

    def _add_ae_distance(self, pairs, filename):
        ae_dir = ['data/images/image_ae_1.csv',
                  'data/images/image_ae_2.csv',
                  'data/images/image_ae_3.csv',
                  'data/images/image_ae_4.csv',
                  'data/images/image_ae_5.csv',
                  'data/images/image_ae_6.csv',
                  'data/images/image_ae_7.csv',
                  'data/images/image_ae_8.csv',
                  'data/images/image_ae_9.csv']

        # Add pairwise cosine distance based on autoencoder embeddings
        print("Adding ae distance")
        total = len(pairs)
        idx = 0
        with open(filename, 'w') as fout:
            fout.write(','.join(['itemID_1', 'itemID_2',
                                 'ae_mean_distance',
                                 'ae_min_distance',
                                 'ae_max_distance',
                                 'ae_sum_distance',
                                 ])+'\n')
            try:
                for pair in pairs[['itemID_1', 'itemID_2',
                                'images_array_1', 'images_array_2']].astype(str).itertuples():
                    print("%s of %s" % (idx, total))
                    idx += 1

                    # Extract ae vectors
                    img_idx_arr_1 = [
                        int
                        (img)
                        for img in pair.images_array_1.split
                        (',') if img != '-1' and type(img) == str]
                    img_idx_arr_2 = [
                        int
                        (img)
                        for img in pair.images_array_2.split
                        (',') if img != '-1' and type(img) == str]
                    img_ae_1 = []
                    for img_idx in img_idx_arr_1:
                        img_dir_idx = int(str(img_idx)[-2])
                        with open(ae_dir[img_dir_idx-1], 'r') as fin:
                            fin.readline()
                            for line in fin:
                                if int(line.split(',')[0]) == img_idx:
                                    img_ae_1.append(np.fromstring(line.split(',',1)[1][1:-1], sep=','))
                    img_ae_2 = []
                    for img_idx in img_idx_arr_2:
                        img_dir_idx = int(str(img_idx)[-2])
                        with open(ae_dir[img_dir_idx-1], 'r') as fin:
                            fin.readline()
                            for line in fin:
                                if int(line.split(',')[0]) == img_idx:
                                    img_ae_2.append(np.fromstring(line.split(',',1)[1][1:-1], sep=','))
                    img_ae_1 = np.asarray(img_ae_1)
                    img_ae_2 = np.asarray(img_ae_2)

                    if len(img_ae_1) > 0 and len(img_ae_2) > 0:
                        ae_dist  = pairwise_distances(img_ae_1, img_ae_2, metric=cosine_similarity)
                    else:
                        ae_dist = np.array([-1])

                    fout.write(
                        ','.join(
                            [pair.itemID_1, pair.itemID_2,
                                str(ae_dist.mean()),
                                str(ae_dist.min()),
                                str(ae_dist.max()),
                                str(ae_dist.sum())])
                        + '\n'
                    )
            except Exception as e:
                print e
                import pdb; pdb.set_trace()  # XXX BREAKPOINT

    def _add_phash_distance(self, pairs, filename):
        phash = pd.read_csv('data/images/image_hash_phash.csv', index_col=0)

        # Add pairwise phash distance for pair of image arrays
        print("Adding phash distance")
        total = len(pairs)
        idx = 0
        with open(filename, 'w') as fout:
            fout.write(','.join(['itemID_1', 'itemID_2',
                                 'phash_mean_distance',
                                 'phash_min_distance',
                                 'phash_max_distance',
                                 'phash_sum_distance',
                                 ])+'\n')
            for pair in pairs[['itemID_1', 'itemID_2',
                               'images_array_1', 'images_array_2']].astype(str).itertuples():
                print("%s of %s" % (idx, total))
                idx += 1
                try:
                    try:
                        phash_1 = phash.loc[
                            [int
                             (img)
                             for img in
                             pair.images_array_1.split
                             (',')
                             if img != '-1' and type(img)
                             == str]].dropna().image_hash.tolist()
                        phash_1 = np.array(
                            [list(hash) for hash in phash_1]).view(
                            np.uint8)
                    except KeyError:
                        phash_1 = np.array([])
                    try:
                        phash_2 = phash.loc[
                            [int
                             (img)
                             for img in
                             pair.images_array_2.split
                             (',')
                             if img != '-1' and type(img)
                             == str]].dropna().image_hash.tolist()
                        phash_2 = np.array(
                            [list(hash) for hash in phash_2]).view(
                            np.uint8)
                    except KeyError:
                        phash_2 = np.array([])

                    if len(phash_1) > 0 and len(phash_2) > 0:
                        phash_dist = pairwise_distances(
                            phash_1,
                            phash_2,
                            metric=hamming_loss)
                    else:
                        phash_dist = np.array([-1])

                    fout.write(
                        ','.join(
                            [pair.itemID_1, pair.itemID_2,
                             str(phash_dist.mean()),
                             str(phash_dist.min()),
                             str(phash_dist.max()),
                             str(phash_dist.sum())])
                        + '\n'
                    )
                except Exception as e:
                    print(e)
                    import pdb
                    pdb.set_trace()

    def _add_edit_distance(self, pairs, filename):
        # Edit distance of texts
        print("Adding edit distance")
        total = len(pairs)
        idx = 0
        with open(filename, 'w') as fout:
            fout.write(','.join(['itemID_1', 'itemID_2',
                                 'title_token_sort_ratio',
                                 'title_ratio',
                                 'attrsJSON_distance',
                                 'description_ratio',
                                 ])+'\n')
            for pair in pairs[['itemID_1', 'itemID_2',
                               'title_1', 'title_2',
                               'attrsJSON_1', 'attrsJSON_2',
                               'description_1', 'description_2']].astype(str).itertuples():
                print("%s of %s" % (idx, total))
                idx += 1
                fout.write(
                    ','.join(
                        [pair.itemID_1, pair.itemID_2,
                         str(fuzz.token_sort_ratio(pair.title_1,
                                                   pair.title_2)),
                         str(fuzz.ratio(pair.title_1,
                                        pair.title_2)),
                         str(distance(pair.attrsJSON_1,
                                      pair.attrsJSON_2)),
                         str(fuzz.ratio(pair.description_1,
                                        pair.description_2))])
                    + '\n'
                )


def load_data(train_egs=None, test_egs=None):
    """
    Load ItemInfo, do feature engineering on ItemInfo, then merge into ItemPairs,
    do feature engineering on ItemPairs, then return train/test examples and
    features used
    :param train_egs: Randomly sample train_egs examples for training
    :param test_egs: Randomly sample test_egs examples for testing
    :return: train df, test df and list of features
    """
    featurizer = OfflineFeaturizer()

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
    iteminfo_train = pd.read_csv("./data/ItemInfo_train.csv", dtype=types,
                                 usecols=['itemID', 'images_array'])
    iteminfo_train.fillna(-1, inplace=True)

    print("Load ItemInfo_test.csv")
    iteminfo_test = pd.read_csv("./data/ItemInfo_test.csv", dtype=types,
                                usecols=['itemID', 'images_array'])
    iteminfo_test.fillna(-1, inplace=True)

    # Merge iteminfo onto itempairs
    itempairs_train, itempairs_test = featurizer._merge_itempairs(
        iteminfo_train, iteminfo_test, train_egs, test_egs)

    # Feature engineering on ItemPairs, i.e. pair of items
    start_time = time.time()
    itempairs_train = featurizer._add_ae_distance(
       itempairs_train,
       'itempairs_train_ae_distances.csv')
    itempairs_test = featurizer._add_ae_distance(
       itempairs_test,
       'itempairs_test_ae_distances.csv')
    # itempairs_train = featurizer._add_phash_distance(
    #    itempairs_train,
    #    'itempairs_train_phash_distances.csv')
    # itempairs_test = featurizer._add_phash_distance(
    #    itempairs_test,
    #    'itempairs_test_phash_distances.csv')
    # itempairs_train = featurizer._add_edit_distance(
    #    itempairs_train,
    #    'itempairs_train_edit_distances.csv')
    # itempairs_test = featurizer._add_edit_distance(
    #    itempairs_test,
    #    'itempairs_test_edit_distances.csv')
    print(
        'Feature eng ItemPairs time: {} minutes'.format(
            round(
                (time.time() - start_time)/60,
                2)))

if __name__ == '__main__':
    load_data()
