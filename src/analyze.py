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
    if int(chainer.__version__[0]) < 2:
        print("chainer version>=2.0.0 is required.")
        sys.exit()

    ################################
    # Make necessary directories

    if not os.path.exists(constants.LOG_DIR):
        os.mkdir(constants.LOG_DIR)
    if not os.path.exists(constants.MODEL_DIR):
        os.makedirs(constants.MODEL_DIR)

    ################################
    # Get arguments and intialize trainer

    if process_name == 'tagging':
        args = arguments.TaggerArguments().parse_arguments()
        trainer = trainers.TaggerTrainer(args)
    elif process_name == 'dual_tagging':
        args = arguments.DualTaggerArguments().parse_arguments()
        trainer = trainers.DualTaggerTrainer(args)
    elif process_name == 'parsing':
        args = arguments.ParserArguments().parse_arguments()
        trainer = trainers.ParserTrainer(args)
    elif process_name == 'attribute_annotation':
        args = arguments.AttributeAnnotatorArguments().parse_arguments()
        trainer = trainers.AttributeAnnotatorTrainer(args)
    else:
        print('Invalide task name: {}'.format(args.task), file=sys.stderr)
        sys.exit()

    ################################
    # Prepare GPU
    use_gpu = args.gpu >= 0
    if use_gpu:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        chainer.config.cudnn_deterministic = args.use_cudnn

    ################################
    # Load external token unigram (typically word) embedding model

    if args.unigram_embed_model_path and args.execute_mode != 'interactive':
        trainers.load_unigram_embedding_model(args.unigram_embed_model_path)

    ################################
    # Load feature extractor and classifier model

    # self.pretrained_unigram_embed_dim = self.embed_model.wv.syn0[0].shape[0] 
    if args.model_path:
        trainer.load_model(args.model_path)

    elif process_name == 'dual_tagging':
        if args.submodel_path:
            trainer.load_submodel(args.submodel_path)
            trainer.init_hyperparameters(args)
        else:
            print('Error: model_path or sub_model_path must be specified for dual sequence tagging.')
            sys.exit()

    else:
        if args.dic_obj_path:
            trainer.load_dic(args.dic_obj_path)
        trainer.init_hyperparameters(args)
    trainer.init_feat_extractor(use_gpu=use_gpu)

    ################################
    # Initialize dic of strings from external dictionary

    if process_name == 'tagging':
        trainer.load_external_dictionary()

    ################################
    # Load dataset and set up dic

    if args.execute_mode == 'train':
        trainer.load_data_for_training()
    elif args.execute_mode == 'eval':
        trainer.load_data_for_test()

    ################################
    # Set up evaluator, classifier and optimizer

    trainer.setup_evaluator()

    if not trainer.classifier:
        trainer.init_model()
    else:
        trainer.update_model()

    if args.gpu >= 0:
        trainer.classifier.to_gpu()

    trainer.setup_optimizer()

    ################################
    # Run

    if args.execute_mode == 'train':
        trainer.run_train_mode()
    elif args.execute_mode == 'eval':
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
