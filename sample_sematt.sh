################################################################
# Sample commands for sematt task
################################################################


################################################################
# [0] show usage

# python src/seikanlp.py -h


################################################################
# [1.1] sematt: train a new model from the training data

TASK=sematt
TRAIN_DATA=data/fujitsu/work_higashiyama/processed/wt/train_10k.cs
DEVEL_DATA=data/fujitsu/work_higashiyama/processed/wt/eval_7k.cs
INPUT_FORMAT=wl

# python src/seikanlp.py --task $TASK -x train --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT


################################################################
# [1.2] sematt: re-train a model using the trained model from the same or new training data

MODEL=models/main/20180116_1525.pkl

# python src/seikanlp.py --task $TASK -x train --train_data $TRAIN_DATA --devel_data $DEVEL_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.3] sematt: evaluate the trained model on the test data

TEST_DATA=$DEVEL_DATA

# python src/seikanlp.py --task $TASK -x eval --test_data $TEST_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.4] sematt: decode the input text data by the trained model

DECODE_DATA=data/fujitsu/work_higashiyama/processed/wt/sample.wl
INPUT_FORMAT=wl

# python src/seikanlp.py --task $TASK -x decode --decode_data $DECODE_DATA --input_data_format $INPUT_FORMAT --model_path $MODEL


################################################################
# [1.5] sematt: decode interactively the input text by the trained model

python src/seikanlp.py --task $TASK -x interactive --model_path $MODEL
