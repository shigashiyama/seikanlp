################################################################
# Sample commands


################################################################
# [0] Show usage

# python src/segmenter.py -h


# Common arguments

GPU="0 --cudnn"
ITER=2000
EPOCH_END=1
BATCH=100

EMBED_DIM=300
RNN_UNITS=800
RNN_LAYERS=2
DROPOUT=0.1


################################################################
# [1] Train a new model from the training data

TRAIN=data/bccwj_data/MA_SUW/core/train_0.9.tsv
VALID=data/bccwj_data/MA_SUW/core/dev_0.1.tsv
FORMAT="bccwj_seg --subpos_depth 0"

# python src/segmenter.py \
#        -x train \
#        -g $GPU \
#        -i $ITER \
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
MODEL=models/main/save/20171017_1242_i0.694.npz

# python src/segmenter.py \
#        -x train \
#        -g $GPU \
#        -i $ITER \
#        --epoch_begin $EPOCH_BEGIN \
#        -e $EPOCH_END \
#        -b $BATCH \
#        -t $TRAIN \
#        -v $VALID \
#        -m $MODEL 


################################################################
### [3] Evaluate the trained model on the test data

TEST=data/bccwj_data/MA_SUW/core/test_ClassA-1.tsv

# python src/segmenter.py \
#        -q \
#        -x eval \
#        -g $GPU \
#        -b $BATCH \
#        --test_data $TEST \
#        -m $MODEL



################################################################
### [4] Decode the raw text data by the trained model

RAW=data/bccwj_data/MA_SUW/for_kytea/core-tmp_val.tsv 
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
