import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import pickle
import random

class Data:
    def __init__(self):
        self.data_path = ['input/Crimes_-_'+str(i)+'.csv' for i in range(2015, 2017)]
        self.target_name = 'Primary Type'
        self.to_delete = ['ID', 'Case Number', 'Block', 'IUCR',
                          'Description', 'Arrest', 'Beat', 'District',
                          'Ward', 'FBI Code', 'X Coordinate', 'Y Coordinate',
                          'Updated On', 'Location']
        self.data = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.community_name = pickle.load(open('input/community_name.p'))

        self.read_data()
        self.extract_features()

    def read_data(self):
        """
        Read in train and test data
        """
        print 'Read in data...'
        dataset = []
        for path in self.data_path:
            dataset.append(pd.read_csv(path,
                                       # nrows=5000,
                                       index_col='Date',
                                       parse_dates=['Date']))
        self.data = pd.concat(dataset, axis=0)
        self.data = self.data.dropna(axis=0, how='any')
        print self.data.info()

    def random_sample(self, sample_size):
        index = random.sample(np.arange(self.data.shape[0]), int(self.data.shape[0] * sample_size))
        self.data = self.data.iloc[index, :]

    def dummies(self, col, name):
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def keep_major_cat(self, col, top_n):
        counts = self.data[col].value_counts()

        def location(x):
            if x == 'APARTMENT':
                return 'RESIDENCE'
            if x in counts.index[:top_n]:
                return x
            else:
                return 'OTHER'

        def primary_type(x):
            if x in counts.index[:top_n] or x == 'HOMICIDE':
                return x
            else:
                return 'OTHER OFFENSE'

        if col == 'Primary Type':
            self.target_name = counts.index[:top_n]
            self.data[col] = self.data[col].apply(lambda x: primary_type(x))
        elif col == 'Location Description':
            self.data[col] = self.data[col].apply(lambda x: location(x))

    def extract_features(self):
        print '\nExtract features...'
        self.data = self.data.drop(self.to_delete, axis=1)
        self.data['Hour'] = self.data.index.hour
        self.data['Weekday'] = self.data.index.weekday
        self.data['Month'] = self.data.index.month
        self.keep_major_cat('Location Description', 6)
        self.keep_major_cat('Primary Type', 8)
        print self.data.info()

    def make_dummies_and_split(self):
        self.data.index = np.arange(self.data.shape[0])
        # self.dummies('Hour', 'Hour_')
        # self.dummies('Weekday', 'Day_')
        # self.dummies('Month', 'Month_')
        self.dummies('Location Description', 'Loc_')
        # self.dummies('Domestic', 'Domestic_')
        # self.target_name = [x for x in self.data.columns if x[:4] == 'Type']
        # print self.data.info()
        self.split_data()

    def split_data(self):
        self.target = self.data['Primary Type']
        del self.data['Primary Type']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                                self.target,
                                                                                test_size=0.4,
                                                                                random_state=1)
        print 'x_Training set has {} rows, {} columns.\n'.format(*self.x_train.shape)
        print 'x_Test set has {} rows, {} columns.\n'.format(*self.x_test.shape)

    def data_info(self):
        """
        Info of train and test data
        """
        print '\nTrain:\n{}\n'.format('-' * 50)
        self.x_train.info()
        print '\nTrain target:\n{}\n'.format('-' * 50)
        self.y_train.info()

    def data_peek(self):
        """
        Peek at the train and test data
        """
        print '\nTrain:\n{}\n'.format('-'*50)
        print self.x_train.head()
        print '\nTrain target:\n{}\n'.format('-'*50)
        print self.y_train.head()

    def final_result(self, prediction, metrics):
        """
        Report the final model performance with validation data set
        """
        if metrics == 'accuracy':
            accuracy_score(self.y_test, prediction)

