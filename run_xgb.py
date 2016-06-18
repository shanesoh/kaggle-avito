"""
Adapted from 'ZFTurbo: https://kaggle.com/zfturbo'
"""
__author__ = 'Shane Soh'

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import datetime
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time
from data_utils.load_data import load_data
from data_utils.utils import (create_submission, get_importance)


def run_default_test(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 8
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_weight = 1
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
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 100000
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

    print("Saving model...")
    now = datetime.datetime.now()
    filename = 'xgb_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    gbm.save_model(filename + '.model')

    print(
        'Training time: {} minutes'.format(
            round(
                (time.time() - start_time)/60,
                2)))
    return test_prediction.tolist(), score


def run_test():
    train, test, features = load_data()
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    test_prediction, score = run_default_test(
        train, test, features, 'isDuplicate')
    print('Real score = {}'.format(score))
    create_submission(score, test, test_prediction)

def run_cv():
    train, test, features = load_data(train_egs=100000, test_egs=1000)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    cv_params = {'max_depth': [5, 8],
                 'min_child_weight': [1, 3]}
    ind_params = {
        'learning_rate': 0.1,
        'seed': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'silent': True}
    xgb_cv = GridSearchCV(xgb.XGBClassifier(**ind_params),
                          cv_params,
                          scoring='roc_auc', cv=5, n_jobs=2,
                          verbose=1)
    xgb_cv.fit(train[features], train['isDuplicate'])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

if __name__ == '__main__':
    run_test()
