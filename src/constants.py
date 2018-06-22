import numpy as np

global __version
__version__ = 'v0.0.3'

### task

TASK_SEG = 'seg'
TASK_HSEG = 'hseg'
TASK_TAG = 'tag'
TASK_SEGTAG = 'segtag'
TASK_DEP = 'dep'
TASK_TDEP = 'tdep'

### for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'


### for data io

SL_TOKEN_DELIM   = ' '
SL_ATTR_DELIM    = '_'
WL_TOKEN_DELIM   = '\n'
WL_ATTR_DELIM    = '\t'
KEY_VALUE_DELIM  = '='
POS_SEPARATOR    = '-'
COMMENT_SYM      = '#'
ATTR_INFO_DELIM  = ','
ATTR_INFO_DELIM2 = ':'
ATTR_INFO_DELIM3 = '_'
ATTR_DELIM_TXT   = '#ATTRIBUTE_DELIMITER'
WORD_CLM_TXT     = '#WORD_COLUMN'
HEAD_CLM_TXT     = '#HEAD_COLUMN'
ARC_CLM_TXT      = '#ARC_COLUMN'
ATTR_CLM_TXT     = '#ATTR_COLUMN'

ATTR_INDEX    = 'index'
ATTR_CHUNKING = 'chunking'
#ATTR_NONNEG   = 'non-negative'
#ATTR_DEPTH    = 'depth'

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
