################################################################
# Sample commands for dual_seg task
################################################################


################################################################
# [0.1] seg: train a new short unit model from the training data

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
TRAIN_DATA=data/bccwj/data/SUW/core/train.tsv
DEVEL_DATA=data/bccwj/data/SUW/core/devel.tsv
INPUT_FORMAT=wl

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --unigram_embed_dim $UNI_EMBED_DIM --rnn_bidirection --rnn_n_layers $RNN_N_LAYERS --rnn_n_units $RNN_N_UNITS --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT


################################################################
# [1.1] dual_seg: train a new long unit model from the training data using short unit model (sub model)

GPU=0
EPOCH_END=1
BREAK_POINT=2000
BATCH=100
LR=1.0
OPTIMIZER=sgd

DROPOUT=0.3

TASK=dual_seg
TRAIN_DATA=data/bccwj/data/SUW/core/train.tsv
DEVEL_DATA=data/bccwj/data/SUW/core/devel.tsv
INPUT_FORMAT=wl
SUBMODEL=models/main/20180111_1527_e0.039.npz
SUBMODEL_USAGE=add

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --submodel_path $SUBMODEL --submodel_usage $SUBMODEL_USAGE


################################################################
# [1.2] dual_seg: re-train a model using the trained model from the same or new training data

EPOCH_BEGIN=2
EPOCH_END=3
MODEL=models/main/20180124_1353_e0.039.npz

# python src/seikanlp.py --task $TASK -x train -g $GPU --cudnn --epoch_begin $EPOCH_BEGIN --epoch_end $EPOCH_END --break_point $BREAK_POINT --batch_size $BATCH --optimizer $OPTIMIZER --learning_rate $LR --rnn_dropout $DROPOUT --mlp_dropout $DROPOUT --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.3] dual_seg: evaluate the trained model on the test data

TEST_DATA=data/bccwj/data/SUW/core/test.tsv

# python src/seikanlp.py --task $TASK -x eval -g $GPU --cudnn --batch_size $BATCH --test_data $TEST_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.4] dual_seg: decode the input text data by the trained model

DECODE_DATA=data/bccwj/data/raw/sample_1k.txt
OUTPUT_FORMAT=sl

# python src/seikanlp.py --task $TASK -x decode -g $GPU --cudnn --batch_size $BATCH --decode_data $DECODE_DATA --output_data_format $OUTPUT_FORMAT --model_path $MODEL


################################################################
# [1.5] dual_seg: decode interactively the input text by the trained model

# python src/seikanlp.py --task $TASK -x interactive -g $GPU --cudnn --output_data_format $OUTPUT_FORMAT --model_path $MODEL
