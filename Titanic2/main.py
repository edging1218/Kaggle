from data import Data
from config import read_config
from feature_engineer import FeatureEngineer
from model import Model
from stack import Stacker


if __name__ == '__main__':
    # read in data and have a peak
    data = Data()
    data.data_peek()

    # read in configuration
    param = read_config()

    # create features
    features = FeatureEngineer(data, param['features'])
    features.feature_info()

    # create model
    model_xgb = Model(features, 'xgb', param['model'])
    # pred = model_xgb.run_all()
    # model_xgb.cross_validation_all(5, 'accuracy')
    model_rf = Model(features, 'rf', param['model'])
    # # model_rf.cross_validation_all(5, 'accuracy', False, True)
    # # model_rf.grid_search('accuracy', 5)
    model_knn32 = Model(features, 'knn32', param['model'])
    # # pred = model_xgb.grid_search('accuracy', 5)
    model_knn128 = Model(features, 'knn128', param['model'])
    # # model_knn128.cross_validation_all(5, 'accuracy', False, True)
    model_logit = Model(features, 'logit', param['model'])
    # model_logit.cross_validation_all(5, 'accuracy', False, True)

    # model_logit.grid_search('accuracy', 5)
    model_svm = Model(features, 'svm', param['model'])
    # #model_svm.grid_search('accuracy', 5)
    #
    model_list = [model_xgb,
                  model_rf,
                  model_knn32,
                  model_knn128,
                  model_logit,
                  model_svm]

    stack_model_xgb = Model(features, 'xgb', param['stack_model'])
    stack_model_svm = Model(features, 'svm', param['stack_model'])
    stack_model_logit = Model(features, 'logit', param['stack_model'])
    stack_model_list = [stack_model_xgb, stack_model_svm, stack_model_logit]

    # stack_model_list = [stack_model_xgb]
    stack = Stacker(features, 5, model_list, stack_model_list, 2)
    # pred = stack.stack_model_grid_search(0)
    # stack.create_cluster_feature(3, ['Age', 'Fare'])
    pred = stack.make_prediction(1)
    # write submission file
    data.write_submission(pred, 'stack_svm_c04.csv')

