import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Stacker:
    def __init__(self, features, k_fold, models, stack_models):
        self.models = models
        self.k_fold = k_fold
        self.features = features
        self.stack_models = stack_models
        self.meta_train = np.zeros((features.ftrain_size[0], len(models)))
        self.meta_test = np.zeros((features.ftest_size[0], len(models)))
        self.create_meta_feature()
        self.res = {}
        self.model_stacking()

    def create_meta_feature(self):
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature(self.k_fold, 'accuracy')
            self.meta_train[:, idx:idx+1] = ptrain
            self.meta_test[:, idx:idx+1] = ptest
        print 'Meta_feature created with shape {} in train and {} in test'.format(self.meta_train.shape, self.meta_test.shape)

    def model_stacking(self):
        for model in self.stack_models:
            assert isinstance(model.model_type, object)
            self.res[model.model_type] = model.run(x=self.meta_train,
                                              y=self.features.data.train_target,
                                              test=self.meta_test)

    def print_res(self):
        print self.res.keys()

