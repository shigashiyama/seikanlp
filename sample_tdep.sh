################################################################
# Sample commands for tdep task
################################################################


################################################################
# [1.1] tdep: train a new model from the training data

GPU=0
EPOCH_END=1
BREAK_POINT=2000
BATCH=100
OPTIMIZER=sgd
LR=1.0

UNI_EMBED_DIM=300
POS_EMBED_DIM=100

RNN_N_LAYERS=3
RNN_N_UNITS=800
DROPOUT=0.3
HIDDEN_MLP_N_UNITS=500
PRED_MLP_N_LAYERS=2
PRED_MLP_N_UNITS=500

TASK=tdep
TRAIN_DATA=
DEVEL_DATA=
INPUT_FORMAT=wl

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --unigram_embed_dim $UNI_EMBED_DIM --rnn_bidirection --rnn_n_layers $RNN_N_LAYERS --rnn_n_units $RNN_N_UNITS --mlp4arcrep_n_units $HIDDEN_MLP_N_UNITS --mlp4labelrep_n_units $HIDDEN_MLP_N_UNITS --mlp4labelpred_n_layers $PRED_MLP_N_LAYERS --mlp4labelpred_n_units $PRED_MLP_N_UNITS --rnn_dropout $DROPOUT --pred_layers_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT


################################################################
# [1.2] tdep: re-train a model using the trained model from the same or new training data

EPOCH_BEGIN=2
EPOCH_END=3
MODEL=

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_begin $EPOCH_BEGIN --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --rnn_dropout $DROPOUT --pred_layers_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.3] tdep: evaluate the trained model on the test data

TEST_DATA=

# python src/seikanlp.py --task $TASK -x eval -g $GPU --cudnn --batch_size $BATCH --test_data $TEST_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.4] tdep: decode the input text data by the trained model

DECODE_DATA=
INPUT_FORMAT=sl

# python src/seikanlp.py --task $TASK -x decode -g $GPU --cudnn --batch_size $BATCH --decode_data $DECODE_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.5] tdep: decode interactively the input text by the trained model

# python src/seikanlp.py --task $TASK -x interactive -g $GPU --cudnn --model_path $MODEL
