import numpy as np


### common

SEG = 'seg'
TAG = 'tag'
SEG_TAG = 'seg_tag'


### for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'


### for data io

SL_TOKEN_DELIM  = ' '
SL_ATTR_DELIM   = '/'
WL_TOKEN_DELIM  = '\n'
WL_ATTR_DELIM   = '\t'
KEY_VALUE_DELIM = '='
POS_SEPARATOR   = '-'
COMMENT_SYM     = '#'
ATTR_DELIM_TXT  = '#ATTRIBUTE_DELIMITER'
WORD_CLM_TXT    = '#WORD_COLUMN'
POS_CLM_TXT     = '#POS_COLUMN'
HEAD_CLM_TXT    = '#HEAD_COLUMN'
ARC_CLM_TXT     = '#ARC_COLUMN'
ATTR_CLM_TXT    = '#ATTR_COLUMN'


### for dictionary

UNK_SYMBOL      = '<UNK>'
NUM_SYMBOL      = '<NUM>'
NONE_SYMBOL     = '<NONE>'
ROOT_SYMBOL     = '<ROOT>'
# PADDING_SYMBOL = '<PAD>'
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
