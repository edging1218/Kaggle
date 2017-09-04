import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class Data():
    def __init__(self):
        self.train_path = 'input/train.csv'
        self.test_path = 'input/test.csv'
        self.output_path = 'submission/submission.csv'
        self.output_id = 'ImageId'
        self.output_name = 'Label'
        self.target = 'label'
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

    def pca(self, n_component):
        pca = PCA(n_components=n_component)
        self.train = pca.fit_transform(self.train)
        self.test = pca.transform(self.test)
        
    def data_size(self):
        """
        Output train and test data size
        """
        print 'Train size is {}.\nTest size is {}'.format(self.train_size, self.test_size)

    def write_submission(self, pred, filename='submission.csv'):
        """
        Write submission file in train and test data
        """
        self.output_path = 'output/' + filename
        idx = np.arange(1, self.test_size[0]+1).astype(int)
        my_solution = pd.DataFrame(pred, idx, columns=[self.output_name])
        my_solution.to_csv(self.output_path, index_label=[self.output_id])
