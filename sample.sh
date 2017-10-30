################################################################
# Sample commands
################################################################


################################################################
# [0] Show usage

# python src/segmenter.py -h


################################################################
# [1] Train a new model from the training data

GPU="0 --cudnn"
EVAL_SIZE=2000
EPOCH_END=1
BATCH=100

EMBED_DIM=300
RNN_UNITS=800
RNN_LAYERS=2
DROPOUT=0.1

TRAIN=
VALID=
FORMAT="wl_seg_tag --subpos_depth 1"

# python src/segmenter.py \
#        -x train \
#        -g $GPU \
#        --evaluation_size $EVAL_SIZE \
#        -e $EPOCH_END \
#        -b $BATCH \
#        -d $EMBED_DIM \
#        -u $RNN_UNITS \
#        -l $RNN_LAYERS \
#        --rnn_bidirection \
#        --dropout $DROPOUT \
#        -t $TRAIN \
#        -v $VALID \
#        -f $FORMAT
       
       
################################################################
### [2] Re-train a model using the trained model from the same or new training data

EPOCH_BEGIN=2
EPOCH_END=3
MODEL=

# python src/segmenter.py \
#        -x train \
#        -g $GPU \
#        --evaluation_size $EVAL_SIZE \
#        --epoch_begin $EPOCH_BEGIN \
#        -e $EPOCH_END \
#        -b $BATCH \
#        -t $TRAIN \
#        -v $VALID \
#        -m $MODEL 


################################################################
### [3] Evaluate the trained model on the test data

TEST=

# python src/segmenter.py \
#        -q \
#        -x eval \
#        -g $GPU \
#        -b $BATCH \
#        --test_data $TEST \
#        -m $MODEL



################################################################
### [4] Decode the raw text data by the trained model

RAW=
OUTPUT=out/out.txt

python src/segmenter.py \
       -q \
       -x decode \
       -g $GPU \
       -b $BATCH \
       -r $RAW \
       -m $MODEL \
       -o $OUTPUT \


################################################################
### [5] Evaluate the learned model on the test data

# python src/segmenter.py \
#        -x interactive \
#        -g $GPU \
#        -b $BATCH \
#        -m $MODEL \
