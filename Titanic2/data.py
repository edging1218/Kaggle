import pandas as pd
import numpy as np


class Data():
    def __init__(self):
        self.train_path = 'input/train.csv'
        self.test_path = 'input/test.csv'
        self.output_path = 'submission/submission.csv'
        self.output_id = 'PassengerId'
        self.target = 'Survived'
        self.train = None
        self.test = None
        self.train_target = None
        self.train_size = None
        self.test_size = None
        self.read_data()

    def read_data(self):
        """
        Read in train and test data
        """
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        # Get response variable
        self.train_target = self.train[self.target]
        del self.train[self.target]
        self.train_size = self.train.shape
        self.test_size = self.test.shape

    def data_info(self):
        """
        Info of train and test data
        """
        print '\nTrain:\n{}\n'.format('-' * 50)
        self.train.info()
        print '\nTest:\n{}\n'.format('-' * 50)
        self.test.info()


    def data_peek(self):
        """
        Peek at the train and test data
        """
        print '\nTrain:\n{}\n'.format('-'*50)
        print self.train.head()
        print '\nTest:\n{}\n'.format('-'*50)
        print self.test.head()

    def write_submission(self, pred, filename='submission.csv'):
        """
        Write submission file in train and test data
        """
        self.output_path = 'submission/' + filename
        idx = np.array(self.test[self.output_id]).astype(int)
        my_solution = pd.DataFrame(pred, idx, columns=[self.target])
        my_solution.to_csv(self.output_path, index_label=[self.output_id])
