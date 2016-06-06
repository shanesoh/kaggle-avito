"""
Adapted from 'ZFTurbo: https://kaggle.com/zfturbo'
"""
__author__ = 'Shane Soh'

from data_utils.load_data import load_data

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time
import datetime
from operator import itemgetter


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def run_default_test(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 8
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print(
        'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(
            eta,
            max_depth,
            subsample,
            colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 260
    early_stopping_rounds = 20
    test_size = 0.1

    X_train, X_valid = train_test_split(
        train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(
        params,
        dtrain,
        num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=True)

    print("Validating...")
    check = gbm.predict(
        xgb.DMatrix(
            X_valid[features]),
        ntree_limit=gbm.best_ntree_limit)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(
        xgb.DMatrix(
            test[features]),
        ntree_limit=gbm.best_ntree_limit)

    print(
        'Training time: {} minutes'.format(
            round(
                (time.time() - start_time)/60,
                2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + \
        str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('id,probability\n')
    total = 0
    for id in test['id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

if __name__ == '__main__':
    train, test, features = load_data(train_egs=100000, test_egs=10000)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    # Get only subset of data
    if 0:
        len_old = len(train.index)
        train = train.sample(frac=0.5)
        len_new = len(train.index)
        print('Reduce train from {} to {}'.format(len_old, len_new))
    test_prediction, score = run_default_test(
        train, test, features, 'isDuplicate')
    print('Real score = {}'.format(score))
    create_submission(score, test, test_prediction)
