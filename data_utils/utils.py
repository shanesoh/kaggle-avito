import datetime
from operator import itemgetter
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold


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


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'generated_submissions/submission_' + \
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


def stack_models(train, test, y, clfs, n_folds=3, scaler=False, eval_fit=False,
                 only_test=False):
    """
    Uses n classifiers to predict on data and generate n meta-features
    :param train: train data
    :param test: test data
    :param y: target labels
    :param clfs: list of classifiers
    :param n_folds: number of folds
    :param scaler: boolean indicating if input features should be normalized
    :params eval_fit: boolean indicating if model should be eval'd while fitting
    """
    num_class = 1
    skf = list(StratifiedKFold(y, n_folds))
    if scaler:
        scaler = preprocessing.StandardScaler().fit(train)
        train_sc = scaler.transform(train)
        test_sc = scaler.transform(test)
    else:
        train_sc = train
        test_sc = test

    # Number of training data x Number of classifiers
    blend_train = np.zeros((train.shape[0], num_class*len(clfs)))

    # Number of testing data x Number of classifiers
    blend_test = np.zeros((test.shape[0], num_class*len(clfs)))

    for j, clf in enumerate(clfs):
        if not only_test:
            print ("\nTraining classifier %s" % (j))
            for i, (tr_index, cv_index) in enumerate(skf):
                print ("\nStacking fold %s of train data" % (i))
                if type(train) == np.ndarray:
                    X_train = train[tr_index]
                    X_cv = train[cv_index]
                else:
                    X_train = train.iloc[tr_index]
                    X_cv = train.iloc[cv_index]
                Y_train = y[tr_index]
                y_cv = y[cv_index]
                if scaler:
                    scaler_cv = preprocessing.StandardScaler().fit(X_train)
                    X_train = scaler_cv.transform(X_train)
                    X_cv = scaler_cv.transform(X_cv)
                if eval_fit:
                    clf.fit(X_train, Y_train, X_cv, y_cv)
                else:
                    clf.fit(X_train, Y_train)
                pred = clf.predict_proba(X_cv)
                if pred.ndim > 1 and pred.shape[1] == 2:
                    pred = pred[:,1]
                blend_train[cv_index, j] = pred.squeeze()

        print("\nStacking test data")
        clf.fit(train_sc, y)
        pred = clf.predict_proba(test_sc)
        if pred.ndim > 1 and pred.shape[1] == 2:
            pred = pred[:,1]
        blend_test[:, j] = pred.squeeze()

    return blend_train, blend_test
