import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score, log_loss
import pandas as pd
import numpy as np


class Model:
    def __init__(self, features, model_type, model_param):
        self.features = features
        self.model_type = model_type
        self.model_param = model_param
        if model_type not in model_param:
            raise Exception('Model cannot be found in the file.')
        self.param = model_param[model_type]
        self.grid_param = None
        self.model = None

    def param_model(self):
        if self.model_type == 'xgb':
            self.model = xgboost.XGBClassifier(**self.param)
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier(**self.param)
        elif self.model_type == 'knn32':
            self.model = KNeighborsClassifier(**self.param)

    def simple_model(self):
        if self.model_type == 'xgb':
            self.model = xgboost.XGBClassifier()
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier()

    def fit_model(self, x, y):
        self.model.fit(x, y)

    def predict_model(self, x_test, probability=False):
        if probability:
            return self.model.predict_proba(x_test)
        else:
            return self.model.predict(x_test)[:, np.newaxis]

    def run(self, x, y, test, pred_prob=False):
        self.param_model()
        self.fit_model(x, y)
        return self.predict_model(test, pred_prob)

    def cross_validation(self, x, y, k_fold, metrics, pred_prob=False):
        x = np.array(x)
        test_pred = pd.DataFrame()
        kf = KFold(n_splits=k_fold, random_state=10)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index, :], x[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            pred_fold = pd.DataFrame(self.run(x_train, y_train, x_test, pred_prob),
                                     index=test_index).add_suffix('_' + self.model_type)
            test_pred = test_pred.append(pred_fold)
        self.calc_metrics(metrics, y, test_pred)
        return test_pred

    def grid_search(self, metric, k_fold):
        print 'Start grid search for {}'.format(self.model_type)
        grid_name = self.model_type + '_grid'
        if grid_name not in self.model_param:
            raise Exception('Parameters for grid search is not available in config.')
        self.grid_param = self.model_param[grid_name]
        self.simple_model()
        if metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif metric == 'logloss':
            scorer = make_scorer(log_loss)
        elif metric == 'fbeta':
            scorer = make_scorer(fbeta_score)
        else:
            print 'Score method not implemented yet.'
            return False
        kf = KFold(n_splits=k_fold, random_state=10)
        grid_obj = GridSearchCV(self.model,
                                param_grid=self.grid_param,
                                scoring=scorer,
                                cv=kf.split(self.features.train_feature))
        grid_fit = grid_obj.fit(self.features.train_feature,
                                self.features.data.train_target)
        print pd.DataFrame(grid_obj.cv_results_)
        print 'Best parameters chosen is:'
        print grid_fit.best_params_
        best_clf = grid_fit.best_estimator_
        best_pred = best_clf.predict(self.features.test_feature)
        # return {'model': best_clf,
        #        'params': grid_fit.best_params_,
        #        'score': grid_fit.best_score_,
        #        'prediction': best_pred}
        return best_pred

    def stacking_feature(self, k_fold, metrics):
        print 'Start feature stacking for {}'.format(self.model_type)
        meta_feature_train = self.cross_validation(self.features.train_feature,
                                                   self.features.data.train_target,
                                                   k_fold,
                                                   metrics,
                                                   True)
        meta_feature_test = self.run(self.features.train_feature,
                                     self.features.data.train_target,
                                     self.features.test_feature,
                                     True)
        return meta_feature_train, meta_feature_test

    def calc_metrics(self, metrics, y_true, y_pred):
        if metrics == 'accuracy':
            print 'accuracy: %f' % (accuracy_score(y_true, y_pred))
        elif metrics == 'logloss':
            y_true_dummy = pd.get_dummies(y_true)
            print 'logloss: %f' % (log_loss(y_true_dummy, y_pred))

    @property
    def get_model_name(self):
        return self.model_type






    #def kfold_cross_validation(self):


