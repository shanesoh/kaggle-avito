from data_utils.load_data import load_data
from data_utils.utils import create_submission
from models.classifiers import NNClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
import datetime


def run_test():
    clf = NNClassifier(
        batch_norm=True,
        hidden_units=512,
        hidden_layers=5,
        dropout=0.5,
        prelu=True,
        hidden_activation='relu',
        batch_size=1024,
        nb_epoch=50,
        optimizer='rmsprop',
        learning_rate=0.001,
        momentum=None,
        decay=None,
        rho=0.9,
        epsilon=1e-08,
        patience=10)

    train, test, features = load_data()
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))

    X_train, X_valid = train_test_split(
        train, test_size=.1, random_state=0)
    y_train = X_train['isDuplicate'].values
    y_valid = X_valid['isDuplicate'].values

    clf.fit(X_train[features], y_train, X_valid[features], y_valid)

    pred_valid = clf.predict_proba(X_valid[features])
    score = roc_auc_score(y_valid, pred_valid)
    print('Check error value: {:.6f}'.format(score))

    print("Predicting on test...")
    pred_test = clf.predict_proba(test[features])

    print("Saving model...")
    now = datetime.datetime.now()
    filename = 'keras_' + \
        str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.model'
    clf.model.save_weights(filename)
    print("Model saved as %s" % filename)

    print("Creating submission...")
    create_submission(score, test, pred_test.tolist())

if __name__ == '__main__':
    run_test()
