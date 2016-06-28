from data_utils.utils import stack_models, create_submission
from data_utils.load_data import load_data
from models.classifiers import NNClassifier
from models.classifiers import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn import preprocessing
import numpy as np
import datetime
import pickle
import click


def stack_knn(train, test, y, only_test=False):
    clf1=KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    clf2=KNeighborsClassifier(n_neighbors=12, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    clf3=KNeighborsClassifier(n_neighbors=24, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    clfs = [clf1, clf2, clf3]

    train_probs, test_probs = stack_models(
        train, test, y, clfs, n_folds=3, scaler=True, eval_fit=False,
        only_test=only_test)

    return train_probs, test_probs

def stack_nn(train, test, y, only_test=False):
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
        hidden_units=1024,
        hidden_layers=8,
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
        train, test, y, clfs, n_folds=3, scaler=True, eval_fit=True,
        only_test=only_test)

    return train_probs, test_probs


def stack_trees(train, test, y, only_test=False):
    clf1 = XGBClassifier(
        eta=.1,
        max_depth=8,
        subsample=.8,
        colsample_bytree=.8,
        min_child_weight=1,
        num_rounds=5000,
        early_stopping_rounds=20)

    clf2 = XGBClassifier(
        eta=.1,
        max_depth=5,
        subsample=.6,
        colsample_bytree=.6,
        min_child_weight=1,
        num_rounds=5000,
        early_stopping_rounds=20)

    clfs = [clf1, clf2]

    train_probs, test_probs = stack_models(
        train, test, y, clfs, n_folds=3, scaler=True, eval_fit=True,
        only_test=only_test)

    return train_probs, test_probs


def get_distances(data, preds):
    preds = preds.mean(axis=1)
    data_class_0 = data[preds <= .5]
    data_class_1 = data[preds > .5]
    dist = []
    for c in [data_class_0, data_class_1]:
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(c)
        z, _ = neigh.kneighbors(data, n_neighbors=5, return_distance=True)
        d1 = z[:, 0]
        d2 = z[:, 0] + z[:, 1] + z[:, 2]
        d3 = z[:, 0] + z[:, 1] + z[:, 2] + z[:, 3] + z[:, 4]
        d = np.vstack((d1, d2, d3)).T
        dist.append(d)
    return np.hstack(dist)


@click.command()
@click.option(
    '--train_egs',
    default=None,
    type=int,
    help='Number of train examples to use')
@click.option(
    '--test_egs',
    default=None,
    type=int,
    help='Number of test examples to use')
@click.option('--load_features', is_flag=True,  help='Load saved features')
def run_test(train_egs, test_egs, load_features):
    train, test, features = load_data(train_egs=train_egs, test_egs=test_egs)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    X_train, X_valid = train_test_split(
        train, test_size=.1, random_state=0)
    y_train = X_train['isDuplicate'].values
    y_valid = X_valid['isDuplicate'].values
    X_train = X_train[features]
    X_valid = X_valid[features]
    X_test = test[features]
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)
    print('X_valid shape: ', X_valid.shape)
    pca = PCA(n_components=24)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)

    # Stack first level models and get meta features
    if not load_features:
        first_level_dir = 'saved_models/first_level/'
        print('Stacking train knn')
        meta_knn_train, meta_knn_test = stack_knn(
            X_train_pca,
            X_test_pca,
            y_train)
        pickle.dump(meta_knn_train, open(
            first_level_dir + 'meta_knn_train.output', 'w'))
        pickle.dump(meta_knn_test, open(
            first_level_dir + 'meta_knn_test.output', 'w'))
        print('Stacking train nn')
        meta_nn_train, meta_nn_test = stack_nn(
            X_train,
            X_test,
            y_train)
        pickle.dump(meta_nn_train, open(
            first_level_dir + 'meta_nn_train.output', 'w'))
        pickle.dump(meta_nn_test, open(
            first_level_dir + 'meta_nn_test.output', 'w'))
        print('Stacking train trees')
        meta_trees_train, meta_trees_test = stack_trees(
            X_train,
            X_test,
            y_train)
        pickle.dump(meta_trees_train, open(
            first_level_dir + 'meta_trees_train.output', 'w'))
        pickle.dump(meta_trees_test, open(
            first_level_dir + 'meta_trees_test.output', 'w'))
        print('Stacking valid knn')
        _, meta_knn_valid = stack_knn(
            X_train_pca,
            X_valid_pca,
            y_train, only_test=True)
        pickle.dump(meta_knn_valid, open(
            first_level_dir + 'meta_knn_valid.output', 'w'))
        print('Stacking valid nn')
        _, meta_nn_valid = stack_nn(
            X_train,
            X_valid,
            y_train, only_test=True)
        pickle.dump(meta_nn_valid, open(
            first_level_dir + 'meta_nn_valid.output', 'w'))
        print('Stacking valid trees')
        _, meta_trees_valid = stack_trees(
            X_train,
            X_valid,
            y_train, only_test=True)
        pickle.dump(meta_trees_valid, open(
            first_level_dir + 'meta_trees_valid.output', 'w'))
    else:
        print('Loading first level features')
        first_level_dir = 'saved_models/first_level/'
        meta_nn_train = pickle.load(open(
            first_level_dir + 'meta_nn_train.output', 'r'))
        meta_nn_test = pickle.load(open(
            first_level_dir + 'meta_nn_test.output', 'r'))
        meta_trees_train = pickle.load(open(
            first_level_dir + 'meta_trees_train.output', 'r'))
        meta_trees_test = pickle.load(open(
            first_level_dir + 'meta_trees_test.output', 'r'))
        meta_knn_train = pickle.load(open(
            first_level_dir + 'meta_knn_train.output', 'r'))
        meta_knn_test = pickle.load(open(
            first_level_dir + 'meta_knn_test.output', 'r'))

        meta_nn_valid = pickle.load(open(
            first_level_dir + 'meta_nn_valid.output', 'r'))
        meta_trees_valid = pickle.load(open(
            first_level_dir + 'meta_trees_valid.output', 'r'))
        meta_knn_valid = pickle.load(open(
            first_level_dir + 'meta_knn_valid.output', 'r'))

    # Create second level features from predictions and input features
    print('Creating second level features')
    dist_trees_train = get_distances(X_train, meta_trees_train)
    dist_nn_train = get_distances(X_train, meta_nn_train)
    dist_knn_train = get_distances(X_train, meta_knn_train)

    dist_trees_test = get_distances(X_test, meta_trees_test)
    dist_nn_test = get_distances(X_test, meta_nn_test)
    dist_knn_test = get_distances(X_test, meta_knn_test)

    dist_trees_valid = get_distances(X_valid, meta_trees_valid)
    dist_nn_valid = get_distances(X_valid, meta_nn_valid)
    dist_knn_valid = get_distances(X_valid, meta_knn_valid)

    train_nn2 = np.hstack(
        [meta_nn_train, meta_trees_train, meta_knn_train,
         dist_nn_train, dist_trees_train, dist_knn_train])
    test_nn2 = np.hstack(
        [meta_nn_test, meta_trees_test, meta_knn_test,
         dist_nn_test, dist_trees_test, dist_knn_test])
    valid_nn2 = np.hstack(
        [meta_nn_valid, meta_trees_valid, meta_knn_valid,
         dist_nn_valid, dist_trees_valid, dist_knn_valid])

    # Train second level models
    print('Training second level models')
    scaler = preprocessing.StandardScaler()
    train_nn2 = scaler.fit_transform(train_nn2)
    test_nn2 = scaler.transform(test_nn2)
    valid_nn2 = scaler.transform(valid_nn2)

    clf_nn2 = NNClassifier(
        batch_norm=True,
        hidden_units=1024,
        hidden_layers=5,
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
    filename = 'saved_models/keras_nn2_' + \
        str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.model'
    clf_nn2.model.save_weights(filename)
    print("Model saved as %s" % filename)

    print("Creating submission...")
    create_submission(score, test, preds_nn2.tolist())


if __name__ == '__main__':
    run_test()
