import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class Stacker:
    def __init__(self, features, k_fold, models, stack_models, n_class):
        self.models = models
        self.k_fold = k_fold
        self.features = features
        self.stack_models = stack_models
        self.n_class = n_class
        self.meta_train = np.zeros((features.train_num, len(models) * n_class))
        self.meta_test = np.zeros((features.test_num, len(models) * n_class))
        self.create_meta_feature()
        self.model_stacking()

    def create_meta_feature(self):
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature(self.k_fold, 'logloss')
            self.meta_train[:, idx * self.n_class:(idx+1)*self.n_class] = ptrain
            self.meta_test[:, idx * self.n_class:(idx+1)*self.n_class] = ptest
        print '\nMeta_feature created with shape {} in train and {} in test\n'.\
            format(self.meta_train.shape, self.meta_test.shape)

    def cluster_score(self, data_set, n_cluster):
        clusterer = GaussianMixture(n_components=n_cluster, random_state=1).fit(data_set)
        preds = clusterer.predict(data_set)
        score = silhouette_score(data_set, preds)
        return score

    def cluster_num(self, cluster_n, features):
        for i in cluster_n:
            score = self.cluster_score(self.features.train_feature[features], i)
            print 'n_cluster: {} \t score: {:.4f}'.format(i, score)

    def create_cluster_feature(self, n_cluster, features):
        data_set = pd.concat([self.features.train_feature[features], self.features.test_feature[features]], axis=0)
        clusterer = GaussianMixture(n_components=n_cluster, random_state=1).fit(data_set)
        preds = clusterer.predict(data_set)
        self.meta_train = np.hstack((self.meta_train, preds[:self.features.train_num][:, np.newaxis]))
        self.meta_test = np.hstack((self.meta_test, preds[self.features.train_num:][:, np.newaxis]))

    def model_stacking(self):
        for model in self.stack_models:
            print 'model %s:' % model.model_type
            model.cross_validation(self.meta_train, self.features.data.train_target, 3, 'accuracy')

    def stack_model_grid_search(self, idx):
        model = self.stack_models[idx]
        return model.grid_search(self.meta_train,
                                 self.features.data.train_target,
                                 self.meta_test,
                                 'accuracy',
                                 5,
                                 model.model_param)

    def make_prediction(self, idx):
        model = self.stack_models[idx]
        return model.run(self.meta_train, self.features.data.train_target, self.meta_test)



