import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.base import BaseEstimator
import pandas as pd


class XGBClassifier(BaseEstimator):

    def __init__(self,
                 nthread=-1,
                 eta=.1,
                 gamma=0,
                 max_depth=6,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 silent=1,
                 seed=0,
                 l2_reg=1,
                 l1_reg=0,
                 num_rounds=260,
                 early_stopping_rounds=20):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y, X_valid=None, y_valid=None):
        params = {"objective": "binary:logistic",
                  "booster": "gbtree",
                  "eval_metric": "auc",
                  "eta": self.eta,
                  "gamma": self.gamma,
                  "max_depth": self.max_depth,
                  "min_child_weight": self.min_child_weight,
                  "max_delta_step": self.max_delta_step,
                  "subsample": self.subsample,
                  "silent": self.silent,
                  "colsample_bytree": self.colsample_bytree,
                  "seed": self.seed,
                  "lambda": self.l2_reg,
                  "alpha": self.l1_reg}

        dtrain = xgb.DMatrix(X, y)
        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, y_valid)
            self.model = xgb.train(
                params,
                dtrain,
                self.num_rounds,
                evals=[(dtrain, 'train'), (dvalid, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=True)
        else:
            self.model = xgb.train(
                params,
                dtrain,
                self.num_rounds)
        return self

    def predict_proba(self, X):
        X = xgb.DMatrix(X)
        preds = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        return preds


class NNClassifier(BaseEstimator):

    def __init__(
            self,
            batch_norm,
            hidden_units,
            hidden_layers,
            dropout,
            prelu,
            hidden_activation,
            batch_size,
            nb_epoch,
            optimizer,
            learning_rate,
            momentum,
            decay,
            rho,
            epsilon,
            patience):
        self.batch_norm = batch_norm
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.prelu = prelu
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon
        self.patience = patience
        self.model = None

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.model = Sequential()
        if max(y) <= 1:
            nb_classes = 1
        else:
            y = np_utils.to_categorical(y)
            nb_classes = y.shape[1]

        # First/hidden layers
        first = True
        hidden_layers = self.hidden_layers
        while hidden_layers > 0:
            if first:
                self.model.add(Dense(self.hidden_units,
                                     input_shape=(X.shape[1],),
                                     init='glorot_normal'))
                first = False
            else:
                self.model.add(Dense(self.hidden_units,
                                     init='glorot_normal'))
                if self.batch_norm:
                    self.model.add(BatchNormalization())
                if self.prelu:
                    self.model.add(PReLU())
                else:
                    self.model.add(Activation(self.hidden_activation))
            self.model.add(Dropout(self.dropout))
            hidden_layers -= 1

        # Output layer
        if nb_classes > 1:
            self.model.add(
                Dense(
                    nb_classes,
                    activation='softmax',
                    init='glorot_normal'))
        else:
            self.model.add(Dense(
                1,
                activation='sigmoid',
                init='glorot_normal'))

        # Optimizers
        if nb_classes > 1:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'
        if self.optimizer == "sgd":
            sgd = keras.optimizers.SGD(
                lr=self.learning_rate,
                decay=self.decay,
                momentum=self.momentum,
                nesterov=True)
            self.model.compile(loss=loss, optimizer=sgd)
        if self.optimizer == "rmsprop":
            rmsprop = keras.optimizers.RMSprop(
                self.learning_rate,
                rho=self.rho,
                epsilon=self.epsilon)
            self.model.compile(
                loss=loss,
                optimizer=rmsprop)
        if self.optimizer == "adagrad":
            adagrad = keras.optimizers.Adagrad(
                self.learning_rate,
                epsilon=self.epsilon)
            self.model.compile(
                loss=loss,
                optimizer=adagrad)
        if self.optimizer == "adadelta":
            adadelta = keras.optimizers.Adadelta(
                self.learning_rate,
                rho=self.rho,
                epsilon=self.epsilon)
            self.model.compile(
                loss=loss,
                optimizer=adadelta)
        if self.optimizer == "adam":
            self.model.compile(
                loss=loss,
                optimizer='adam')

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.patience)
        if X_valid is not None and y_valid is not None:
            if type(X) == pd.DataFrame:
                X = X.values
                X_valid = X_valid.values
            self.model.fit(X, y,
                           validation_data=(X_valid, y_valid),
                           nb_epoch=self.nb_epoch,
                           batch_size=self.batch_size,
                           callbacks=[early_stopping])
        else:
            if type(X) == pd.DataFrame:
                X = X.values
            self.model.fit(X, y,
                           nb_epoch=self.nb_epoch,
                           batch_size=self.batch_size,
                           callbacks=[early_stopping])
        return self

    def predict_proba(self, X):
        if type(X) == pd.DataFrame:
            X = X.values
        preds = self.model.predict_proba(X)
        preds = preds.squeeze()
        return preds
