from data import Data
from config import read_config
from feature_engineer import FeatureEngineer
from model import Model
from stack import Stacker


if __name__ == '__main__':
    # read in data and have a peak
    data = Data()
    data.peek()

    # read in configuration
    param = read_config()

    # create features
    features = FeatureEngineer(data, param['features'])
    features.print_feature()

    # create model
    model_xgb = Model(features, 'xgb', param['model'])
    # pred = model.grid_search('accuracy', 5)

    model_list = [model_xgb]

    stack_model_xgb = Model(features, 'xgb', param['stack_model'])
    stack_model_list = [stack_model_xgb]

    stack = Stacker(features, 3, model_list, stack_model_list)
    stack.print_res()

    # write submission file
    # data.write_submission(pred)

