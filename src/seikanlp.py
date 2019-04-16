## chainer v3 is required

import sys
import os

import chainer
from chainer import cuda

import constants


class SeikaNLP(object):
    # to be implemented in sub-class
    def get_args(self):
        return None


    # to be implemented in sub-class
    def get_trainer(self, args):
        return None


    def run(self):
        ################################
        # Make necessary directories

        if not os.path.exists(constants.LOG_DIR):
            os.mkdir(constants.LOG_DIR)
            if not os.path.exists(constants.MODEL_DIR):
                os.makedirs(constants.MODEL_DIR)

        ################################
        # Get arguments and intialize trainer

        args = self.get_args()
        trainer = self.get_trainer(args)

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

        if args.model_path:
            trainer.load_model()
        else:
            trainer.init_hyperparameters()

        ################################
        # Setup feature extractor and initialize dic from external dictionary
    
        trainer.init_feature_extractor(use_gpu)
        trainer.load_external_dictionary()

        ################################
        # Load dataset and set up dic

        if args.execute_mode == 'train':
            trainer.load_training_and_development_data()
        elif args.execute_mode == 'eval':
            trainer.load_test_data()
        elif args.execute_mode == 'decode':
            trainer.load_decode_data()
        elif args.execute_mode == 'interactive':
            trainer.setup_data_loader()

        ################################
        # Set up classifier

        if not trainer.classifier:
            trainer.init_model()
        else:
            trainer.update_model(train=args.execute_mode=='train')

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
