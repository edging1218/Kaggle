import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score
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

    def simple_model(self):
        if self.model_type == 'xgb':
            self.model = xgboost.XGBClassifier()
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier()

    def fit_model(self, x, y):
        self.model.fit(x, y)

    def predict_model(self, x_test):
        return self.model.predict(x_test)[:, np.newaxis]

    def run(self, x, y, test):
        self.param_model()
        self.fit_model(x, y)
        return self.predict_model(test)

    def grid_search(self, metric, k_fold):
        print 'Start grid search for {}'.format(self.model_type)
        grid_name = self.model_type + '_grid'
        if grid_name not in self.model_param:
            raise Exception('Parameters for grid search is not available in config.')
        self.grid_param = self.model_param[grid_name]
        self.simple_model()
        if metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
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
        meta_feature_train = np.zeros((self.features.ftrain_size[0], 1))
        kf = KFold(n_splits=k_fold, random_state=10)
        for train_index, test_index in kf.split(self.features.train_feature):
            x_train, x_test = self.features.train_feature.iloc[train_index, :], self.features.train_feature.iloc[test_index, :]
            y_train, y_test = self.features.data.train_target[train_index], self.features.data.train_target[test_index]
            self.param_model()
            self.fit_model(x_train, y_train)
            test_pred = self.predict_model(x_test)
            meta_feature_train[test_index, :] = test_pred
            self.calc_metrics(metrics, y_test, test_pred)
        meta_feature_test = self.run(self.features.train_feature, self.features.data.train_target, self.features.test_feature)
        return meta_feature_train, meta_feature_test

    def calc_metrics(self, metrics, y_true, y_pred):
        if metrics == 'accuracy':
            print accuracy_score(y_true, y_pred)

    @property
    def get_model_name(self):
        return self.model_type






    #def kfold_cross_validation(self):


