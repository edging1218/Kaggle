import yaml
import pprint

def read_config():
    """
    Read in configuration
    """
    paths = 'config/config.yml'
    with open(paths, 'r') as f:
        config = yaml.load(f)
    pprint.pprint(config)
    return config