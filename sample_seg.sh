################################################################
# Sample commands for seg task
################################################################


################################################################
# [1.1] seg: train a new model from the training data

GPU=0
EPOCH_END=1
BREAK_POINT=2000
BATCH=100
LR=1.0
OPTIMIZER=sgd

UNI_EMBED_DIM=300
RNN_N_LAYERS=2
RNN_N_UNITS=800
DROPOUT=0.3

TASK=seg
TRAIN_DATA=
DEVEL_DATA=
INPUT_FORMAT=wl

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --unigram_embed_dim $UNI_EMBED_DIM --rnn_bidirection --rnn_n_layers $RNN_N_LAYERS --rnn_n_units $RNN_N_UNITS --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT -q


################################################################
# [1.2] seg: re-train a model using the trained model from the same or new training data

EPOCH_BEGIN=2
EPOCH_END=3
MODEL=

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_begin $EPOCH_BEGIN --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.3] seg: evaluate the trained model on the test data

TEST_DATA=

# python src/seikanlp.py --task $TASK -x eval -g $GPU --cudnn --batch_size $BATCH --test_data $TEST_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.4] seg: decode the input text data by the trained model

DECODE_DATA=
OUTPUT_FORMAT=sl

# python src/seikanlp.py --task $TASK -x decode -g $GPU --cudnn --batch_size $BATCH --decode_data $DECODE_DATA --output_data_format $OUTPUT_FORMAT --model_path $MODEL


################################################################
# [1.5] seg: decode interactively the input text by the trained model

# python src/seikanlp.py --task $TASK -x interactive -g $GPU --cudnn --output_data_format $OUTPUT_FORMAT --model_path $MODEL


################################################################
# [2.1] seg: train a new model with pre-trained word embedding model

EMBED_MODEL=

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --learning_rate $LR --unigram_embed_dim $UNI_EMBED_DIM --rnn_bidirection --rnn_n_layers $RNN_N_LAYERS --rnn_n_units $RNN_N_UNITS --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT  --unigram_embed_model_path $EMBED_MODEL --fix_pretrained_embed


################################################################
# [2.2] seg: decode the input text by the trained model with pre-trained word embedding model

# python src/seikanlp.py --task $TASK -x decode -g $GPU --cudnn --batch_size $BATCH --decode_data $DECODE_DATA --output_data_format $OUTPUT_FORMAT --model_path $MODEL --unigram_embed_model_path $EMBED_MODEL


################################################################
# [3.1] seg: train a new model with dictionary features

EXTERNAL_DIC=
FEATURE_TMP=default

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --learning_rate $LR --unigram_embed_dim $UNI_EMBED_DIM --rnn_bidirection --rnn_n_layers $RNN_N_LAYERS --rnn_n_units $RNN_N_UNITS --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --external_dic_path $EXTERNAL_DIC --feature_template $FEATURE_TMP


################################################################
# [3.2] seg: decode interactively the input text by the trained model with dictionary features

# python src/seikanlp.py --task $TASK -x decode -g $GPU --cudnn --batch_size $BATCH --decode_data $DECODE_DATA --output_data_format $OUTPUT_FORMAT --model_path $MODEL

