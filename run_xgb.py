"""
Adapted from 'ZFTurbo: https://kaggle.com/zfturbo'
"""
__author__ = 'Shane Soh'

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import datetime
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from data_utils.load_data import load_data
from data_utils.utils import (create_submission, get_importance)
from models.classifiers import XGBClassifier


def run_test():
    clf = XGBClassifier(
        eta = .1,
        max_depth = 8,
        subsample = .8,
        colsample_bytree = .8,
        min_child_weight = 1,
        num_rounds = 100000,
        early_stopping_rounds = 20)

    train, test, features = load_data(train_egs=1000, test_egs=100)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    X_train, X_valid = train_test_split(
        train, test_size=.1, random_state=0)
    y_train = X_train['isDuplicate'].values
    y_valid = X_valid['isDuplicate'].values

    clf.fit(X_train[features], y_train, X_valid[features], y_valid)

    print("Validating...")
    pred_valid = clf.predict_proba(X_valid[features])
    score = roc_auc_score(y_valid, pred_valid)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(clf.model, features)
    print('Importance array: ', imp)

    print("Predicting on test...")
    pred_test = clf.predict_proba(test[features])

    print("Saving model...")
    now = datetime.datetime.now()
    filename = 'xgb_' + \
        str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.model'
    clf.model.save_model(filename)
    print("Model saved as %s" % filename)

    print("Creating submission...")
    create_submission(score, test, pred_test.tolist())


def run_cv():
    train, test, features = load_data(train_egs=1000, test_egs=100)
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
    import pdb
    pdb.set_trace()  # XXX BREAKPOINT

if __name__ == '__main__':
    run_test()
