import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re


class FeatureEngineer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.train_feature = pd.DataFrame()
        self.test_feature = pd.DataFrame()
        self.feature_names = config.keys()
        data_set = pd.concat([self.data.train, self.data.test], axis=0)
        self.train_num = self.data.train.shape[0]
        self.test_num = self.data.test.shape[0]
        feature = self.create_features(data_set)
        self.train_feature = feature[:self.train_num]
        self.test_feature = feature[self.train_num:]

    def create_features(self, data_set):
        """
        engineer specified features in config and add to train/test_feature
        """
        feature = pd.DataFrame()
        for f, status in self.config.items():
            if status == 'good':
                feature[f] = data_set[f]
            elif status == 'fillna_scale':
                feature[f] = data_set[f]
                mean = self.data.train[f].mean()
                feature[f] = feature[f].fillna(mean)
                feature[f] = feature[f]/mean
            elif status == 'fit_transform':
                encoder = LabelEncoder()
                feature[f] = encoder.fit_transform(data_set[f])
            elif status == 'get_dummies':
                new_train = pd.get_dummies(data_set[f])
                feature = pd.concat([feature, new_train], axis=1)
            elif status == 'first_letter':
                new_train = pd.get_dummies(data_set[f].apply(lambda x: str(x)[0]))
                new_train.columns = [coln + f for coln in new_train.columns]
                feature = pd.concat([feature, new_train], axis=1)
            elif status == 'title':
                name = data_set[f].apply(lambda x: re.split('\W+', x)[1])
                name[name == 'Mr'] = 1
                name[(name == 'Miss') | (name == 'Mlle')] = 2
                name[(name == 'Mrs') | (name == 'Mme')] = 3
                name[name == 'Master'] = 4
                name[(name != 1) & (name != 2) & (name != 3) & (name != 4)] = 0
                feature[f] = pd.Series(name, dtype=int)
        return feature


    def feature_info(self):
        """
        Print the info of features
        """
        print self.train_feature.info()
        print self.test_feature.info()

    def feature_peek(self):
        """
        Take a peek of the features
        """
        print self.train_feature.head()
        print self.test_feature.head()


