"""
Experiment with various doc2vec models and save best performing one
"""
import numpy as np
import pandas as pd
import re
import logging
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim import utils

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


class DescIter(object):

    def __init__(self):
        types = {
            'itemID': np.dtype(int),
            'description': np.dtype(str),
        }
        print("Load ItemInfo_train.csv")
        self.iteminfo_train = pd.read_csv(
            "./data/ItemInfo_train.csv",
            usecols=types.keys(),
            dtype=types)
        self.iteminfo_train.fillna(-1, inplace=True)

        print("Load ItemInfo_test.csv")
        self.iteminfo_test = pd.read_csv(
            "./data/ItemInfo_test.csv",
            usecols=types.keys(),
            dtype=types)
        self.iteminfo_test.fillna(-1, inplace=True)

    def _tokenize(self, text):
        text = str(text).lower()
        text = re.sub('\d', 'DG', text)
        text = utils.to_unicode(text)
        return text.split()

    def __iter__(self):
        for data in [self.iteminfo_train, self.iteminfo_test]:
            for row in data.itertuples():
                yield TaggedDocument(words=self._tokenize(row.description),
                                     tags=[int(row.itemID)])


if __name__ == '__main__':
    desc = DescIter()

    print("Training d2v model")
    model = Doc2Vec(
        desc,
        alpha=0.025,
        min_alpha=.001,
        min_count=5,
        window=8,
        size=100,
        sample=1e-4,
        negative=5,
        workers=4)
    model.save('./description.d2v')
    import pdb
    pdb.set_trace()  # XXX BREAKPOINT
