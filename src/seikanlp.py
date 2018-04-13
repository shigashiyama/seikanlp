import sys
import os
import copy
import argparse

#os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer
from chainer import cuda

import common
import constants
import arguments
from trainers import tagger_trainer, parser_trainer, attribute_annotator_trainer


def run():
    # if int(chainer.__version__[0]) < 3:
    #     print("chainer version>=3.0.0 is required.")
    #     sys.exit()

    ################################
    # Make necessary directories

    if not os.path.exists(constants.LOG_DIR):
        os.mkdir(constants.LOG_DIR)
    if not os.path.exists(constants.MODEL_DIR):
        os.makedirs(constants.MODEL_DIR)

    ################################
    # Get arguments and intialize trainer

    args = arguments.SelectiveArgumentParser().parse_args()

    if common.is_single_st_task(args.task):
        if args.task == constants.TASK_HSEG:
            trainer = tagger_trainer.HybridSegmenterTrainer(args)
        else:
            trainer = tagger_trainer.TaggerTrainer(args)
    elif common.is_dual_st_task(args.task):
        trainer = tagger_trainer.DualTaggerTrainer(args)
    elif common.is_parsing_task(args.task):
        trainer = parser_trainer.ParserTrainer(args)
    elif common.is_attribute_annotation_task(args.task):
        trainer = attribute_annotator_trainer.AttributeAnnotatorTrainer(args)

    ################################
    # Prepare GPU
    use_gpu = 'gpu' in args and args.gpu >= 0
    if use_gpu:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        chainer.config.cudnn_deterministic = args.use_cudnn

    ################################
    # Load external embedding model

    trainer.load_external_embedding_models()

    ################################
    # Load classifier model

    if (args.model_path or 
        'submodel_path' in args and args.submodel_path):
        trainer.load_model()
    else:
        # if args.dic_obj_path:
        #     trainer.load_dic(args.dic_obj_path)
        trainer.init_hyperparameters()

    ################################
    # Setup feature extractor and initialize dic from external dictionary
    
    trainer.init_feature_extractor(use_gpu)
    trainer.load_external_dictionary()

    ################################
    # Load dataset and set up dic

    if args.execute_mode == 'train':
        trainer.load_data_for_training()
    elif args.execute_mode == 'eval':
        trainer.load_test_data()
    elif args.execute_mode == 'decode':
        trainer.load_decode_data()

    ################################
    # Set up classifier

    if not trainer.classifier:
        trainer.init_model()
    # else:
    #     trainer.update_model()

    if use_gpu:
        trainer.classifier.to_gpu()

    ################################
    # Run

    if args.execute_mode == 'train':
        trainer.setup_optimizer()
        trainer.setup_evaluator()
        trainer.run_train_mode()

    elif args.execute_mode == 'eval':
        trainer.setup_evaluator()
        trainer.run_eval_mode()

    elif args.execute_mode == 'decode':
        trainer.run_decode_mode()

    elif args.execute_mode == 'interactive':
        trainer.run_interactive_mode()

    ################################
    # Terminate

    trainer.close()


if __name__ == '__main__':
    run()
