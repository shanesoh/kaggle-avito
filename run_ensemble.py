from data_utils.utils import stack_models, create_submission
from data_utils.load_data import load_data
from models.classifiers import NNClassifier
from models.classifiers import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import numpy as np
import datetime

def stack_nn(train, test, y):
    clf1 = NNClassifier(
        batch_norm=True,
        hidden_units=1024,
        hidden_layers=5,
        dropout=0.5,
        prelu=True,
        hidden_activation='relu',
        batch_size=1024,
        nb_epoch=1000,
        optimizer='rmsprop',
        learning_rate=0.001,
        momentum=None,
        decay=None,
        rho=0.9,
        epsilon=1e-08,
        patience=8)

    clf2 = NNClassifier(
        batch_norm=True,
        hidden_units=512,
        hidden_layers=5,
        dropout=0.5,
        prelu=False,
        hidden_activation='relu',
        batch_size=1024,
        nb_epoch=1000,
        optimizer='adam',
        learning_rate=0.001,
        momentum=None,
        decay=None,
        rho=0.9,
        epsilon=1e-08,
        patience=8)

    clfs = [clf1, clf2]

    train_probs, test_probs = stack_models(
        train, test, y, clfs, n_folds=3, scaler=True, eval_fit=True)

    return train_probs, test_probs


def stack_trees(train, test, y):
    clf1 = XGBClassifier(
        eta=.1,
        max_depth=8,
        subsample=.8,
        colsample_bytree=.8,
        min_child_weight=1,
        num_rounds=500,
        early_stopping_rounds=20)

    clf2 = XGBClassifier(
        eta=.1,
        max_depth=6,
        subsample=.8,
        colsample_bytree=.8,
        min_child_weight=1,
        num_rounds=500,
        early_stopping_rounds=20)

    clfs = [clf1, clf2]

    train_probs, test_probs = stack_models(
        train, test, y, clfs, n_folds=3, scaler=True, eval_fit=True)

    return train_probs, test_probs


def run_test():
    train, test, features = load_data(train_egs=100000)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    X_train, X_valid = train_test_split(
        train, test_size=.1, random_state=0)
    y_train = X_train['isDuplicate'].values
    y_valid = X_valid['isDuplicate'].values

    # Stack first level models and get meta features
    print('Stacking models')
    meta_nn_train, meta_nn_test = stack_nn(
        X_train[features],
        test[features],
        y_train)
    meta_trees_train, meta_trees_test = stack_trees(
        X_train[features],
        test[features],
        y_train)
    meta_nn_train, meta_nn_valid = stack_nn(
        X_train[features],
        X_valid[features],
        y_train)
    meta_trees_train, meta_trees_valid = stack_trees(
        X_train[features],
        X_valid[features],
        y_train)

    # Train second level models
    print('Training second level models')
    train_nn2 = np.hstack([meta_nn_train, meta_trees_train])
    test_nn2 = np.hstack([meta_nn_test, meta_trees_test])
    valid_nn2 = np.hstack([meta_nn_valid, meta_trees_valid])

    clf_nn2 = NNClassifier(
        batch_norm=True,
        hidden_units=256,
        hidden_layers=2,
        dropout=0.5,
        prelu=True,
        hidden_activation='relu',
        batch_size=1024,
        nb_epoch=1000,
        optimizer='adam',
        learning_rate=0.001,
        momentum=None,
        decay=None,
        rho=0.9,
        epsilon=1e-08,
        patience=8)

    clf_nn2.fit(train_nn2, y_train, valid_nn2, y_valid)

    pred_valid = clf_nn2.predict_proba(valid_nn2)
    score = roc_auc_score(y_valid, pred_valid)
    print('AUC Score: {:.6f}'.format(score))

    print("Predicting on test...")
    preds_nn2 = clf_nn2.predict_proba(test_nn2)

    print("Saving model...")
    now = datetime.datetime.now()
    filename = 'keras_nn2_' + \
        str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.model'
    clf_nn2.model.save_weights(filename)
    print("Model saved as %s" % filename)

    print("Creating submission...")
    create_submission(score, test, preds_nn2.tolist())


if __name__ == '__main__':
    run_test()
