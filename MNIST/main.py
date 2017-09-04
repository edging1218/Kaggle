from data import Data
import numpy as np
from model import Model


if __name__ == '__main__':
    data = Data()
    # data.data_info()

    data.pca(75)
    knn8 = Model(data, 'knn8', {'knn8': {'n_neighbors': 8}})
    # knn8.cross_validation_all(4, 'accuracy')
    pred = knn8.run_all()
    # pred.to_csv('temp.csv')
    data.write_submission(pred, 'pca75knn8.csv')

