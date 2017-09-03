import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.train_feature = pd.DataFrame()
        self.test_feature = pd.DataFrame()
        self.feature_names = []
        self.ftrain_size = [0, 0]
        self.ftest_size = [0, 0]
        self.create_features()

    def create_features(self):
        """
        engineer specified features in config and add to train/test_feature
        """
        for f, status in self.config.items():
            self.feature_names.append(f)
            if status == 'good':
                self.train_feature[f] = self.data.train[f]
                self.test_feature[f] = self.data.test[f]
            elif status == 'fillna':
                self.train_feature[f] = self.data.train[f]
                self.test_feature[f] = self.data.test[f]
                mean = self.data.train[f].mean()
                self.train_feature[f] = self.train_feature[f].fillna(mean)
                self.test_feature[f] = self.test_feature[f].fillna(mean)
            elif status == 'fit_transform':
                encoder = LabelEncoder()
                self.train_feature[f] = encoder.fit_transform(self.data.train[f])
                self.test_feature[f] = encoder.fit_transform(self.data.test[f])
            elif status == 'get_dummies':
                new_train = pd.get_dummies(self.data.train[f])
                new_test = pd.get_dummies(self.data.test[f])
                self.train_feature = pd.concat([self.train_feature, new_train], axis=1)
                self.test_feature = pd.concat([self.test_feature, new_test], axis=1)
        self.ftrain_size = self.train_feature.shape
        self.ftest_size = self.test_feature.shape

    def print_feature(self):
        """
        Print
        """
        print self.train_feature.info()
        print self.test_feature.info()


