import numpy as np


### common

SEG = 'seg'
TAG = 'tag'
SEG_TAG = 'seg_tag'


### for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'


### for data io

DELIM1_SYMBOL       = '_'
DELIM2_SYMBOL       = '\t'
KEY_VALUE_SEPARATOR = '='
POS_SEPARATOR       = '-'
COMMENT_SYM         = '#'
DELIM_TXT           = '#DELIMITER'
WORD_CLM_TXT        = '#WORD_COLUMN'
POS_CLM_TXT         = '#POS_COLUMN'
HEAD_CLM_TXT        = '#HEAD_COLUMN'
ARC_CLM_TXT         = '#ARC_COLUMN'


### for dictionary

UNK_SYMBOL      = '<UNK>'
NUM_SYMBOL      = '<NUM>'
#PADDING_SYMBOL = '<PAD>'
ROOT_SYMBOL     = '<ROOT>'
DUMMY_SYMBOL    = '<DUM>' # dummy parent node symbol for children that do not have parent
SEG_LABELS      = 'BIES'


### for parsing

NO_PARENTS_ID = -2       # indicate that parent of the word is nothing
UNK_PARENT_ID = -3       # indicate that parent of the word is unknown


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
