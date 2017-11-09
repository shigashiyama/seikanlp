import sys
import os

import chainer
from chainer import cuda

import constants
import arguments
import trainers

def run(process_name):
    if chainer.__version__[0] != '2':
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
    # Load feature extractor and classifier model

    if args.model_path:
        trainer.load_model(args.model_path)
        id2token_org = copy.deepcopy(trainer.indices.token_indices.id2str)
        id2label_org = copy.deepcopy(trainer.indices.xxx_indices.id2str) # TODO modify
    else:
        trainer.init_hyperparameters(args)
        id2token_org = {}
        id2label_org = {}
    trainer.init_feat_extractor(use_gpu=use_gpu)

    ################################
    # Load word embedding model

    # TODO modify for parsing
    if args.unit_embed_model_path and args.execute_mode != 'interactive':
        embed_model = emb.read_model(args.unit_embed_model_path)
        embed_dim = embed_model.wv.syn0[0].shape[0]
        if embed_dim != hparams['embed_dimension']:
            print('Given embedding model is discarded due to dimension confriction')
            embed_model = None
    else:
        embed_model = None

    ################################
    # Initialize indices of strings from external dictionary

    if process_name == 'tagging' and args.dict_path and not trainer.indices:
        dic_path = args.dict_path
        trainer.indices = dictionary.load_dictionary(dic_path, read_pos=False)
        trainer.log('Load dictionary: {}'.format(dic_path))
        trainer.log('Vocab size: {}'.format(len(trainer.indices.token_indices)))

        if not dic_path.endswith('pickle'):
            base = dic_path.split('.')[0]
            dic_pic_path = base + '.pickle'
            with open(dic_pic_path, 'wb') as f:
                pickle.dump(trainer.indices, f)
            trainer.log('Dumpped dictionary: {}'.format(dic_pic_path))

    ################################
    # Load dataset and set up indices

    if args.execute_mode == 'train':
        trainer.load_data_for_training(embed_model)
    elif args.execute_mode == 'eval':
        trainer.load_data_for_test(embed_model)

    ################################
    # Set up classifier and optimizer

    if not trainer.classifier:
        trainer.init_model()

    grow_vocab = id2token_org and (len(trainer.indices.token_indices.id2str) > len(id2token_org))
    if embed_model or grow_vocab:
        trainer.classifier.grow_embedding_layer(
            trainer.indices.token_indices.id2str, id2token_org, embed_model)
    # TODO modify
    # grow_labels = id2label_org and (len(trainer.classifier.indices.id2str) > len(id2label_org))
    # if grow_labels:
    #     trainer.classifier.grow_inference_layers(trainer.classifier.indices.id2label, id2label_org)

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
