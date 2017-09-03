import numpy as np


class Stacker:
    def __init__(self, features, k_fold, models, stack_models, n_class):
        self.models = models
        self.k_fold = k_fold
        self.features = features
        self.stack_models = stack_models
        self.n_class = n_class
        self.meta_train = np.zeros((features.ftrain_size[0], len(models) * n_class))
        self.meta_test = np.zeros((features.ftest_size[0], len(models) * n_class))
        self.create_meta_feature()
        self.model_stacking()

    def create_meta_feature(self):
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature(self.k_fold, 'logloss')
            self.meta_train[:, idx * self.n_class:(idx+1)*self.n_class] = ptrain
            self.meta_test[:, idx * self.n_class:(idx+1)*self.n_class] = ptest
        print '\nMeta_feature created with shape {} in train and {} in test\n'.format(self.meta_train.shape, self.meta_test.shape)

    def model_stacking(self):
        for model in self.stack_models:
            print 'model %s:' % model.model_type
            model.cross_validation(self.meta_train, self.features.data.train_target, 3, 'accuracy')


