import sys
import os
import copy
import argparse

#os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer
from chainer import cuda

import constants
import arguments
import trainers


def run(process_name):
    if int(chainer.__version__[0]) < 3:
        print("chainer version>=3.0.0 is required.")
        sys.exit()

    ################################
    # Make necessary directories

    if not os.path.exists(constants.LOG_DIR):
        os.mkdir(constants.LOG_DIR)
    if not os.path.exists(constants.MODEL_DIR):
        os.makedirs(constants.MODEL_DIR)

    ################################
    # Get arguments and intialize trainer

    args = arguments.SelectiveArgumentParser().parse_args()

    if process_name == 'tagging':
        trainer = trainers.TaggerTrainer(args)
    elif process_name == 'dual_tagging':
        trainer = trainers.DualTaggerTrainer(args)
    elif process_name == 'parsing':
        trainer = trainers.ParserTrainer(args)
    elif process_name == 'attribute_annotation':
        trainer = trainers.AttributeAnnotatorTrainer(args)
    else:
        print('Error: invalid process name: {}'.format(process_name), file=sys.stderr)
        sys.exit()

    ################################
    # Prepare GPU
    use_gpu = 'gpu' in args and args.gpu >= 0
    if use_gpu:                 # TODO confirm
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        chainer.config.cudnn_deterministic = args.use_cudnn

    ################################
    # Load external token unigram (typically word) embedding model

    if ('unigram_embed_model_path' in args and args.unigram_embed_model_path):
        trainer.load_external_embedding_model(args.unigram_embed_model_path)

    ################################
    # Load classifier model

    if args.model_path:
        trainer.load_model(args.model_path)

    elif 'submodel_path' in args: # TODO confirm
        if args.submodel_path:
            trainer.load_submodel(args.submodel_path)
            trainer.init_hyperparameters(args)
        else:
            print('Error: model_path or sub_model_path must be specified for {}'.format(args.task))
            sys.exit()

    else:
        # if args.dic_obj_path:
        #     trainer.load_dic(args.dic_obj_path)
        trainer.init_hyperparameters(args)

    ################################
    # Setup feature extractor and initialize dic from external dictionary
    
    trainer.init_feat_extractor(use_gpu=use_gpu)
    if 'external_dic_path' in args: #TODO confirm
        trainer.load_external_dictionary()

    ################################
    # Load dataset and set up dic

    if args.execute_mode == 'train':
        trainer.load_data_for_training()
    elif args.execute_mode == 'eval':
        trainer.load_data_for_test()
    elif args.execute_mode == 'decode':
        trainer.load_decode_data()

    ################################
    # Set up classifier

    if not trainer.classifier:
        trainer.init_model()
    else:
        trainer.update_model()

    if 'gpu' in args and args.gpu >= 0:
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
