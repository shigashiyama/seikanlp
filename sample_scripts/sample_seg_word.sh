## Common parameters

GPU_ID=0
TASK=seg
TAGGING_UNIT=hybrid
BATCH=30
WORD_EMBED_MODEL= #/path/to/word_embedding_model


################################################
# Train a segmentation model

MODE=train
EP_BEGIN=1
EP_END=20
BREAK_POINT=30
LR_DECAY='10:1:0.9'
CHAR_EMB_DIM=300
WORD_EMB_DIM=300
RNN_LAYERS=2
RNN_UNITS=600
RNN2_LAYERS=1
RNN2_UNITS=600
RNN_DROPOUT=0.4
WORD_VEC_DROPOUT=0.4
COMPOSITION_TYPE=con
TRAIN_DATA=data/sample/text1.seg.sl
DEVEL_DATA=data/sample/text2.seg.sl
INPUT_FORMAT=sl

# python src/seika_tagger.py \
#        --task $TASK \
#        --tagging_unit $TAGGING_UNIT \
#        --execute_mode $MODE \
#        --gpu $GPU_ID --cudnn \
#        --epoch_begin $EP_BEGIN \
#        --epoch_end $EP_END \
#        --break_point $BREAK_POINT \
#        --batch_size $BATCH \
#        --sgd_lr_decay $LR_DECAY \
#        --unigram_embed_dim $CHAR_EMB_DIM \
#        --chunk_embed_dim $WORD_EMB_DIM \
#        --rnn_bidirection \
#        --rnn_n_layers $RNN_LAYERS \
#        --rnn_n_units $RNN_UNITS \
#        --rnn_n_layers2 $RNN2_LAYERS \
#        --rnn_n_units2 $RNN2_UNITS \
#        --rnn_dropout $RNN_DROPOUT \
#        --chunk_vector_dropout $WORD_VEC_DROPOUT \
#        --pretrained_embed_usage init \
#        --chunk_pooling_type $COMPOSITION_TYPE \
#        --train_data $TRAIN_DATA \
#        --devel_data $DEVEL_DATA \
#        --input_data_format $INPUT_FORMAT \
#        # --chunk_embed_model_path $WORD_EMBED_MODEL \

MODEL= #models/main/20XXXXXX_XXXX_eXX.XXX.npz


################################################
# Evaluate the learned model

MODE=eval
TEST_DATA=data/sample/text2.seg.sl
INPUT_FORMAT=sl

# python src/seika_tagger.py \
#        --task $TASK \
#        --tagging_unit $TAGGING_UNIT \
#        --execute_mode $MODE \
#        --gpu $GPU_ID --cudnn \
#        --batch_size $BATCH \
#        --test_data $TEST_DATA \
#        --input_data_format $INPUT_FORMAT \
#        --model_path $MODEL \
#        --quiet \
#        # --chunk_embed_model_path $WORD_EMBED_MODEL \


################################################
# Segment a raw text by the learned model

MODE=decode
DECODE_DATA=data/sample/text3.txt
OUTPUT_FORMAT=sl

# python src/seika_tagger.py \
#        --task $TASK \
#        --tagging_unit $TAGGING_UNIT \
#        --execute_mode $MODE \
#        --gpu $GPU_ID --cudnn \
#        --batch_size $BATCH \
#        --decode_data $DECODE_DATA \
#        --model_path $MODEL \
#        --output_data_format $OUTPUT_FORMAT \
#        --quiet \
#      # --chunk_embed_model_path $WORD_EMBED_MODEL \
