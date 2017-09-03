import yaml


def read_config():
    """
    Read in configuration
    """
    paths = 'config/config.yml'
    with open(paths, 'r') as f:
        config = yaml.load(f)
    print config
    return config