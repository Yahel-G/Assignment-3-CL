# ==============================================================================
# Imports
# ==============================================================================
from configparser import ConfigParser
from os import path

cfg = ConfigParser()
config_dir = path.realpath(path.join(path.dirname(path.realpath(__file__))))
config_file_path = path.join(config_dir, 'ww_config.ini')
cfg.read(config_file_path)

# ==============================================================================
# Algorithm Parameters
# ==============================================================================

IMG_SIZE        = cfg['image'].getint('image_size')

BATCH_SIZE      = cfg['model'].getint('batch_size')
EPOCHS          = cfg['model'].getint('epochs')
SEED            = cfg['model'].getint('seed')
VERBOSE         = cfg['model'].getint('verbose')

TRAIN_SIZE      = cfg['process'].getfloat('train_size')
VALID_SIZE      = cfg['process'].getfloat('valid_size')

DATA_PATH       = cfg['paths'].get('data_path')
LABEL_PATH      = cfg['paths'].get('label_path')
MODEL_PATH      = cfg['paths'].get('model_path')
RESULTS_DIR     = cfg['paths'].get('results_dir')

# ==============================================================================
# Exports...
# ==============================================================================
__all__ = [
    'BATCH_SIZE',
    'DATA_PATH',
    'EPOCHS',
    'IMG_SIZE',
    'LABEL_PATH',
    'MODEL_PATH',
    'RESULTS_DIR',
    'SEED',
    'TRAIN_SIZE',
    'VALID_SIZE',
    'VERBOSE',
           ]
# ==============================================================================
