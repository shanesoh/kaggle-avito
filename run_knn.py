from data_utils.load_data import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split


def run_test():
    clf = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None)

    train, test, features = load_data(train_egs=10000, test_egs=1000)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    X_train, X_valid = train_test_split(
        train, test_size=.1, random_state=0)
    y_train = X_train['isDuplicate'].values
    y_valid = X_valid['isDuplicate'].values

    pca = PCA(n_components=32)
    X_train_pca = pca.fit_transform(X_train[features])
    X_valid_pca = pca.transform(X_valid[features])

    clf.fit(X_train_pca, y_train)

    pred_valid = clf.predict_proba(X_valid_pca)
    score = roc_auc_score(y_valid, pred_valid[:,1])
    print('Check error value: {:.6f}'.format(score))

if __name__ == '__main__':
    run_test()
