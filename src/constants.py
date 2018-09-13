import numpy as np

global __version
__version__ = 'v0.0.3'

### task

TASK_SEG = 'seg'
TASK_SEGTAG = 'segtag'
TASK_DUAL_SEG = 'dseg'
TASK_TAG = 'tag'
TASK_DEP = 'dep'
TASK_TDEP = 'tdep'

### for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'


### for data io

NUM_FOR_REPORTING = 100000

SL_TOKEN_DELIM   = ' '
SL_ATTR_DELIM    = '_'
WL_TOKEN_DELIM   = '\n'
WL_ATTR_DELIM    = '\t'
KEY_VALUE_DELIM  = '='
SUBATTR_SEPARATOR    = '-'
COMMENT_SYM      = '#'
ATTR_INFO_DELIM  = ','
ATTR_INFO_DELIM2 = ':'
ATTR_INFO_DELIM3 = '_'

SL_FORMAT = 'sl'
WL_FORMAT = 'wl'


### for dictionary

UNK_SYMBOL      = '<UNK>'
NUM_SYMBOL      = '<NUM>'
NONE_SYMBOL     = '<NONE>'
ROOT_SYMBOL     = '<ROOT>'

CHUNK      = 'chunk'
UNIGRAM    = 'unigram'   
BIGRAM     = 'bigram'
SUBTOKEN   = 'subtoken'
TOKEN_TYPE = 'token_type'
SEG_LABEL  = 'seg_label' 
ARC_LABEL  = 'arc_label'
ATTR_LABEL = 'attr{}_label'.format

TYPE_HIRA        = '<HR>'
TYPE_KATA        = '<KT>'
TYPE_LONG        = '<LG>'
TYPE_KANJI       = '<KJ>'
TYPE_ALPHA       = '<AL>'
TYPE_DIGIT       = '<DG>'
TYPE_SPACE       = '<SC>'
TYPE_SYMBOL      = '<SY>'
TYPE_ASCII_OTHER = '<AO>'

BOS = '<B>'                     # unused
EOS = '<E>'

SEG_LABELS      = 'BIES'
B = 'B'
I = 'I'
E = 'E' 
S = 'S'
O = 'O'

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
