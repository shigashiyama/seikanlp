import sys
import os
import copy

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
    else:
        print('Invalide process name: {}'.format(process_name), file=sys.stderr)
        sys.exit()

    ################################
    # Prepare GPU
    use_gpu = args.gpu >= 0
    if use_gpu:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        chainer.config.cudnn_deterministic = args.use_cudnn

    ################################
    # Load word embedding model

    if args.token_embed_model_path and args.execute_mode != 'interactive':
        embed_model = trainers.load_embedding_model(args.token_embed_model_path)
        pretrained_token_embed_dim = embed_model.wv.syn0[0].shape[0]
    else:
        embed_model = None
        pretrained_token_embed_dim = 0

    ################################
    # Load feature extractor and classifier model

    if args.model_path:
        trainer.load_model(args.model_path, pretrained_token_embed_dim)
        dic_org = copy.deepcopy(trainer.dic)

    elif process_name == 'dual_tagging':
        if args.sub_model_path:
            trainer.load_sub_model(args.sub_model_path, pretrained_token_embed_dim)
            dic_org = copy.deepcopy(trainer.dic)
            trainer.init_hyperparameters(args)
        else:
            print('Error: model_path or sub_model_path must be specified for dual sequence tagging.')
            sys.exit()

    else:
        if args.dic_obj_path:
            trainer.load_dic(args.dic_obj_path)
        trainer.init_hyperparameters(args)
        dic_org = None
    trainer.init_feat_extractor(use_gpu=use_gpu)

    ################################
    # Initialize dic of strings from external dictionary

    # TODO internal and external dictionary
    if process_name == 'tagging' and args.ext_dic_path and not trainer.dic:
        edic_path = args.ext_dic_path
        trainer.dic = dictionary.load_vocabulary(edic_path, read_pos=False)
        trainer.log('Load external dictionary: {}'.format(edic_path))
        trainer.log('Vocab size: {}'.format(len(trainer.dic.token_indices)))

        if not dic_path.endswith('pickle'):
            base = dic_path.split('.')[0]
            dic_pic_path = base + '.pickle'
            with open(dic_pic_path, 'wb') as f:
                pickle.dump(trainer.dic, f)
            trainer.log('Dumpped dictionary: {}'.format(dic_pic_path))

    ################################
    # Load dataset and set up dic

    if args.execute_mode == 'train':
        trainer.load_data_for_training(embed_model)
    elif args.execute_mode == 'eval':
        trainer.load_data_for_test(embed_model)

    ################################
    # Set up evaluator, classifier and optimizer

    trainer.setup_evaluator()

    if not trainer.classifier:
        trainer.init_model(pretrained_token_embed_dim)
        if embed_model:
            trainer.classifier.load_pretrained_embedding_layer(
                trainer.dic, embed_model, finetuning=(not args.fix_pretrained_embed))

    else:
        trainer.classifier.grow_embedding_layers(dic_org, trainer.dic, embed_model,
                                                 train=(args.execute_mode=='train'))
        trainer.classifier.grow_inference_layers(dic_org, trainer.dic)

    trainer.finalize_model()

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
