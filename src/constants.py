import numpy as np


### common

SEG = 'seg'
TAG = 'tag'
SEG_TAG = 'seg_tag'


### for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'
#OUT_DIR = 'out'


### for data io

DELIM1_SYMBOL = '_'
DELIM2_SYMBOL = '\t'
KEY_VALUE_SEPARATOR = '='
POS_SEPARATOR = '-'
COMMENT_SYM = '#'
DELIM_TXT = '#DELIMITER'
WORD_CLM_TXT = '#WORD_COLUMN'
POS_CLM_TXT = '#POS_COLUMN'
DEP_CLM_TXT = '#DEP_COLUMN'
ARC_CLM_TXT = '#ARC_COLUMN'


### for dictionary

UNK_SYMBOL = '<UNK>'
NUM_SYMBOL = '<NUM>'
ROOT_SYMBOL = '<ROOT>'
SEG_LABELS = 'BIES'


### for feature extraction
# example of expected input: 'seg:L:2-3,4-5,6-10'

FEAT_TEMP_DELIM = ':'
FEAT_TEMP_RANGE_DELIM = ','
FEAT_TEMP_RANGE_SYM = '-'
L = 'L'
R = 'R'
I = 'I'
B = 'B'
E = 'E' 
