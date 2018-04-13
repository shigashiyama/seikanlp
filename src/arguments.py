import sys
import argparse

import common
import constants


class SelectiveArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        
    def parse_args(self):
        all_args = self.get_full_parser().parse_args()
        min_args = self.get_minimum_parser(all_args).parse_args()
        return min_args


    def get_full_parser(self):
        parser = argparse.ArgumentParser()

        ### mode options
        parser.add_argument('--execute_mode', '-x', help=
                            'Choose a mode from among \'train\', \'eval\', \'decode\' and \'interactive\'')
        parser.add_argument('--task', '-t', help=
                            'Choose a task from among \'seg\', \'segtag\', \'tag\', \'dual_seg\', '
                            + '\'dual_segtag\', \'dual_tag\', \'dep\', \'tdep\' and \'sematt\'')
        parser.add_argument('--quiet', '-q', action='store_true',
                            help='Do not output log file and serialized model file')

        ### gpu options
        parser.add_argument('--gpu', '-g', type=int, default=-1, help=
                            'GPU device ID (Use CPU if specify a negative value)')
        parser.add_argument('--cudnn', '-c', dest='use_cudnn', action='store_true', help='Use cuDNN')

        ### training parameters
        parser.add_argument('--epoch_begin', type=int, default=1, help=
                            'Conduct training from i-th epoch (Default: 1)')
        parser.add_argument('--epoch_end', '-e', type=int, default=5, help=
                            'Conduct training up to i-th epoch (Default: 5)')
        parser.add_argument('--break_point', type=int, default=10000, help=
                            'The number of instances which'
                            + ' trained model is evaluated and saved at each multiple of (Default: 10000)')
        parser.add_argument('--batch_size', '-b', type=int, default=100, help=
                            'The number of examples in each mini-batch (Default: 100)')
        parser.add_argument('--grad_clip', type=float, default=5.0, help=
                            'Gradient norm threshold to clip (Default: 5.0)')
        parser.add_argument('--weight_decay_ratio', type=float, default=0.0, help=
                            'Weight decay ratio (Default: 0.0)')

        ### optimizer parameters
        parser.add_argument('--optimizer', '-o', default='sgd', help=
                            'Choose optimizing algorithm from among '
                            + '\'sgd\', \'adam\' and \'adagrad\', \'adadelta\' and \'rmsprop\' (Default: sgd)')
        parser.add_argument('--learning_rate', type=float, default=1.0, help=
                            'Initial learning rate for SGD or AdaGrad (Default: 1.0)')
        parser.add_argument('--sgd_lr_decay', default='', help=
                            'Specify conditions on learning rate decay for SGD'
                            + ' by format \'start:width:rate\''
                            + ' where start indicates the number of epochs to start decay,'
                            + ' width indicates the number epochs maintaining the same decayed learning rate,'
                            + ' and rate indicates the decay late to multipy privious learning late')
        parser.add_argument('--sgd_cyclical_lr', default='', help='\'stepsize:min_lr:max_lr\'')

        parser.add_argument('--sgd_momentum_ratio', dest='momentum', type=float, default=0.0, help=
                            'Momentum ratio for SGD (Default: 0.0)')
        parser.add_argument('--adam_alpha', type=float, default=0.001, help='alpha for Adam (Default: 0.001)')
        parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1 for Adam (Default: 0.9)')
        parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 for Adam (Default: 0.999)')
        parser.add_argument('--adadelta_rho', type=float, default=0.95, help='rho for AdaDelta (Default: 0.95)')
        parser.add_argument('--rmsprop_alpha', type=float, default=0.99, help=
                            'alpha for RMSprop (Default: 0.99)')

        ### data paths and related options
        parser.add_argument('--model_path', '-m', default='', help=
                            'npz/pkl file path of the trained model. \'xxx.hyp\' and \'xxx.s2i\' files'
                            + 'are also read simultaneously when you specify \'xxx_izzz.npz/pkl\' file')

        parser.add_argument('--submodel_path', default='', help=
                            'npz file path of the trained short unit model that is necessary '
                            + 'when train new long unit model for dual_seg/dual_segtag/dual_tag task. '
                            + '\'xxx.hyp\' and \'xxx.s2i\' files are also read simultaneously '
                            + 'when you specify \'xxx_izzz.npz/pkl\' file.')
        parser.add_argument('--submodel_usage', default='independent')
        ### parser.add_argument('--dic_obj_path', help='')
        parser.add_argument('--input_data_path_prefix', '-p', dest='path_prefix', help=
                            'Path prefix of input data')
        parser.add_argument('--train_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of training data')
        parser.add_argument('--devel_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of development data')
        parser.add_argument('--test_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of test data')
        parser.add_argument('--decode_data', default='', help=
                            'File path of input text which succeeds \'input_data_path_prefix\'')
        # parser.add_argument('--label_reference_data', default='',
        #                     help='File path succeeding \'input_data_path_prefix\''
        #                     + ' of data with the same format as training data to load pre-defined labels')
        parser.add_argument('--external_dic_path', help=
                            'File path of external word dictionary listing known words')
        parser.add_argument('--feature_template', default='', help=
                            'Use dictionary features based on given feature template file. '
                            + 'Specify \'defualt\' to use default features '
                            + 'or specify the path of castomized template file')
        parser.add_argument('--output_data', default='', help=
                            'File path to output parsed text')
        # parser.add_argument('--dump_train_data', action='store_true',
        #                     help='Dump data specified as \'train_data\''
        #                     + ' using new file name endding with \'.pickle\'')
        parser.add_argument('--unigram_embed_model_path', default='', help=
                            'File path of pretrained model of token (character or word) unigram embedding')
        parser.add_argument('--input_data_format', '-f', default='', help=
                            'Choose format of input data among from \'wl\' and \'sl\'')
        parser.add_argument('--output_data_format', default='', help=
                            'Choose format of output data among from \'wl\' and \'sl\'')
        parser.add_argument('--output_attribute_delimiter_sl', dest='output_attr_delim', 
                            default=constants.SL_ATTR_DELIM, help=
                            'Specify delimiter symbol between attributes (word, pos etc.) for SL format '
                            + 'when output analysis results on decode/interactive mode (Default \'/\')')

        ### options for data pre/post-processing
        parser.add_argument('--subpos_depth', type=int, default=-1, help=
                            'Set positive integer (use POS up to i-th hierarcy), or '
                            + '0 or negative value (use all sub-POS). '
                            + 'POS hierarchy must be indicated by \'-\' like \'NOUN-GENERAL\'')
        parser.add_argument('--lowercase_alphabets',  dest='lowercase', action='store_true', help=
                            'Lowercase alphabets in input text')
        parser.add_argument('--normalize_digits',  action='store_true', help=
                            'Normalize digits by the same symbol in input text')
        parser.add_argument('--ignored_labels', default='', help=''
                            'Sepecify labels to be ignored on evaluation '
                            + 'by format \'label_1,label_2,...,label_N\'')

        ### model parameters
        # common
        parser.add_argument('--token_freq_threshold', type=int, default=1, help=
                            'Token frequency threshold. Tokens whose frequency are lower than'
                            + 'the the threshold are regarded as unknown tokens (Default: 1)')
        parser.add_argument('--token_max_vocab_size', type=int, default=-1, help=
                            'Maximum vocaburaly size. '
                            + 'low frequency tokens are regarded as unknown tokens so that '
                            + 'vocaburaly size does not exceed the specified size so much '
                            + 'if set positive value (Default: -1)')
        parser.add_argument('--unigram_embed_dim', type=int, default=300, help=
                            'The number of dimension of token (character or word) unigram embedding'
                            + '(Default: 300)')
        parser.add_argument('--subtoken_embed_dim', type=int, default=0, help=
                            'The number of dimension of subtoken (usually character) embedding (Default: 0)')

        # neural models
        parser.add_argument('--pretrained_embed_usage', default='none', help='')
        parser.add_argument('--fix_pretrained_embed', action='store_true', help=
                            'Fix paramerters of pretrained token unigram embedding duaring training')

        # sequene tagging and parsing
        parser.add_argument('--embed_dropout', type=float, default=0.0, help=
                            'Dropout ratio for embedding layers (Default: 0.0)')
        parser.add_argument('--rnn_dropout', type=float, default=0.0, help=
                            'Dropout ratio for RNN vertical layers (Default: 0.0)')
        parser.add_argument('--rnn_unit_type', default='lstm', help=
                            'Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain\' (Default: lstm)')
        parser.add_argument('--rnn_bidirection', action='store_true', help='Use bidirectional RNN')
        parser.add_argument('--rnn_n_layers', type=int, default=1, help=
                            'The number of RNN layers (Default: 1)')
        parser.add_argument('--rnn_n_units', type=int, default=800, help=
                            'The number of hidden units of RNN (Default: 800)')

        # sequence tagging
        parser.add_argument('--bigram_freq_threshold', type=int, default=1, help='')
        parser.add_argument('--bigram_max_vocab_size', type=int, default=-1, help='')
        parser.add_argument('--bigram_embed_dim', type=int, default=0, help='')
        parser.add_argument('--bigram_embed_model_path', default='', help='')
        parser.add_argument('--tokentype_embed_dim', type=int, default=0, help='')
        parser.add_argument('--mlp_dropout', type=float, default=0.0, help=
                            'Dropout ratio for MLP of sequence tagging model (Default: 0.0)')
        parser.add_argument('--mlp_n_layers', type=int, default=1, help=
                            'The number of layers of MLP of sequence tagging model (Default: 1)')
        parser.add_argument('--mlp_n_units', type=int, default=300, help=
                            'The number of hidden units of MLP of sequence tagging model (Default: 300)')
        parser.add_argument('--inference_layer_type', dest='inference_layer', default='crf', help=
                            'Choose type of inference layer for sequence tagging model from between '
                            + '\'softmax\' and \'crf\' (Default: crf)')


        # char and word hybrid segmentation
        parser.add_argument('--biaffine_type', default='', help='')
        parser.add_argument('--chunk_loss_ratio', type=float, default=0.0, help='')
        parser.add_argument('--chunk_freq_threshold', type=int, default=1, help='')
        parser.add_argument('--chunk_max_vocab_size', type=int, default=-1, help='')
        parser.add_argument('--rnn_n_layers2', type=int, default=0, help='')
        parser.add_argument('--rnn_n_units2', type=int, default=0, help='')
        parser.add_argument('--chunk_embed_dim', type=int, default=300)
        parser.add_argument('--chunk_embed_model_path', default='', help=
                            'File path of pretrained model of chunk (typically word) embedding')
        parser.add_argument('--chunk_pooling_type', default='ave')
        parser.add_argument('--max_chunk_len', type=int, default=4)
        parser.add_argument('--biaffine_dropout', type=float, default=0.0)
        parser.add_argument('--chunk_vector_dropout', type=float, default=0.0)
        parser.add_argument('--chunk_vector_element_dropout', type=float, default=0.0)
        parser.add_argument('--use_gold_chunk', action='store_true')
        parser.add_argument('--ignore_unknown_pretrained_chunk', action='store_true')

        # parsing
        parser.add_argument('--hidden_mlp_dropout', type=float, default=0.0, help=
                            'Dropout ratio for MLP of parsing model (Default: 0.0)')
        parser.add_argument('--pred_layers_dropout', type=float, default=0.0, help=
                            'Dropout ratio for prediction layers of parsing model (Default: 0.0)')
        parser.add_argument('--pos_embed_dim', type=int, default=100, help=
                            'The number of dimension of pos embedding of parsing model (Default: 100)')
        parser.add_argument('--mlp4arcrep_n_layers', type=int, default=1, help=
                            'The number of layers of MLP of parsing model for arc representation '
                            + '(Defualt: 1)')
        parser.add_argument('--mlp4arcrep_n_units', type=int, default=300, help=
                            'The number of hidden units of MLP of parsing model for arc representation '
                            + '(Defualt: 300)')
        parser.add_argument('--mlp4labelrep_n_layers', type=int, default=1, help=
                            'The number of layers of MLP of parsing model for label representation '
                            + '(Defualt: 1)')
        parser.add_argument('--mlp4labelrep_n_units', type=int, default=300, help=
                            'The number of hidden units of MLP of parsing model for arc representation '
                            + '(Defualt: 300)')
        parser.add_argument('--mlp4labelpred_n_layers', type=int, default=1, help=
                            'The number of layers of MLP of parsing model for label prediction '
                            + '(Defualt: 1)')
        parser.add_argument('--mlp4labelpred_n_units', type=int, default=300, help=
                            'The number of hidden units of MLP of parsing model for label prediction '
                            + '(Defualt: 300)')

        # parser.add_argument('--mlp4pospred_n_layers', type=int, default=0, help='')
        # parser.add_argument('--mlp4pospred_n_units', type=int, default=300, help='')

        return parser


    def get_minimum_parser(self, args):
        parser = argparse.ArgumentParser()

        # common options
        self.add_common_options(parser, args)

        # task dependent options
        if (common.is_single_st_task(args.task) or 
            common.is_dual_st_task(args.task) or
            common.is_parsing_task(args.task)):
            self.add_gpu_options(parser, args)

        elif common.is_attribute_annotation_task(args.task):
            pass
        else:
            print('Error: invalid task name: {}'.format(args.task), file=sys.stderr)
            sys.exit()

        self.add_task_dependent_options(parser, args)

        # mode dependent options
        if args.execute_mode == 'train':
            self.add_train_mode_options(parser, args)
        elif args.execute_mode == 'eval':
            self.add_eval_mode_options(parser, args)
        elif args.execute_mode == 'decode':
            self.add_decode_mode_options(parser, args)
        elif args.execute_mode == 'interactive':
            self.add_interactive_mode_options(parser, args)
        else:
            print('Error: invalid execute mode: {}'.format(args.execute_mode), file=sys.stderr)
            sys.exit()

        return parser


    def add_common_options(self, parser, args):
        # mode options
        parser.add_argument('--execute_mode', '-x', required=True, default=args.execute_mode)
        if args.model_path:
            parser.add_argument('--task', '-t', default=args.task)
        else:
            parser.add_argument('--task', '-t', required=True, default=args.task)
        parser.add_argument('--quiet', '-q', action='store_true', default=args.quiet)

        # options for data pre/post-processing
        parser.add_argument('--lowercase_alphabets',  dest='lowercase', action='store_true',
                                 default=args.lowercase)
        parser.add_argument('--normalize_digits',  action='store_true', default=args.normalize_digits)


    def add_gpu_options(self, parser, args):
        # gpu options
        parser.add_argument('--gpu', '-g', type=int, default=args.gpu)
        parser.add_argument('--cudnn', '-c', dest='use_cudnn', action='store_true', default=args.use_cudnn)


    def add_train_mode_options(self, parser, args):
        # training parameters
        parser.add_argument('--epoch_begin', type=int, default=args.epoch_begin)
        parser.add_argument('--epoch_end', '-e', type=int, default=args.epoch_end)
        parser.add_argument('--break_point', type=int, default=args.break_point)
        parser.add_argument('--batch_size', '-b', type=int, default=args.batch_size)
        parser.add_argument('--ignored_labels', default=args.ignored_labels)

        # data paths and related options
        parser.add_argument('--model_path', '-m', default=args.model_path)
        parser.add_argument('--input_data_path_prefix', '-p', dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--train_data', required=True, default=args.train_data)
        parser.add_argument('--devel_data', default=args.devel_data)
        self.add_input_data_format_option(parser, args)

        # model parameters
        parser.add_argument('--token_freq_threshold', type=int, default=args.token_freq_threshold)
        parser.add_argument('--token_max_vocab_size', type=int, default=args.token_max_vocab_size)

        # optimizer parameters
        parser.add_argument('--grad_clip', type=float, default=args.grad_clip)
        parser.add_argument('--weight_decay_ratio', type=float, default=args.weight_decay_ratio)
        parser.add_argument('--optimizer', '-o', default=args.optimizer)

        if args.optimizer == 'sgd':
            self.add_sgd_options(parser, args)
        elif args.optimizer == 'adam':
            self.add_adam_options(parser, args)
        elif args.optimizer == 'adagrad':
            self.add_adagrad_options(parser, args)
        elif args.optimizer == 'adadelta':
            self.add_adadelta_options(parser, args)
        elif args.optimizer == 'rmsprop':
            self.add_rmsprop_options(parser, args)
        else:
            print('Error: invalid optimizer name: {}'.format(args.optimizer), file=sys.stderr)
            sys.exit()


    def add_eval_mode_options(self, parser, args):
        # evaluation parameters
        parser.add_argument('--batch_size', '-b', type=int, default=args.batch_size)
        parser.add_argument('--ignored_labels', default=args.ignored_labels)

        # data paths and related options
        parser.add_argument('--model_path', '-m', required=True, default=args.model_path)
        parser.add_argument('--input_data_path_prefix', '-p', dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--test_data', required=True, default=args.test_data)
        self.add_input_data_format_option(parser, args)


    def add_decode_mode_options(self, parser, args):
        # decoding parameters
        parser.add_argument('--batch_size', '-b', type=int, default=args.batch_size)

        # data paths and related options
        parser.add_argument('--model_path', '-m', required=True, default=args.model_path)
        parser.add_argument('--input_data_path_prefix', '-p', dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--decode_data', required=True, default=args.decode_data)
        parser.add_argument('--output_data', '-o', default=args.output_data)
        self.add_input_data_format_option(parser, args)
        self.add_output_data_format_options(parser, args)


    def add_interactive_mode_options(self, parser, args):
        parser.add_argument('--model_path', '-m', required=True, default=args.model_path)
        self.add_output_data_format_options(parser, args)



    def add_input_data_format_option(self, parser, args):
        if args.execute_mode == 'train' or args.execute_mode == 'eval':
            if common.is_single_st_task(args.task) or common.is_dual_st_task(args.task):
                if args.input_data_format == 'sl' or args.input_data_format == 'wl':
                    parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

                else:
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified among from {sl, wl}.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()
                    
            else:
                if args.input_data_format == 'wl':
                    parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

                elif not args.input_data_format:
                    parser.add_argument('--input_data_format', '-f', default='wl')

                else:
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified among from {wl}.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()

        else:                   # decode
            if common.is_segmentation_task(args.task):
                pass

            elif args.task == constants.TASK_TAG or args.task == constants.TASK_DUAL_TAG:
                if args.input_data_format == 'sl' or args.input_data_format == 'wl':
                    parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

                else:
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified among from {sl, wl}.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()
                
            else:               # dep, tdep, sematt
                if args.input_data_format == 'sl' or args.input_data_format == 'wl':
                    parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

                elif not args.input_data_format:
                    parser.add_argument('--input_data_format', '-f', default='wl')

                else:
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified among from {sl, wl}.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()
            

    def add_output_data_format_options(self, parser, args):
        if args.output_data_format == 'sl':
            parser.add_argument('--output_data_format', default=args.output_data_format)
            parser.add_argument('--output_attribute_delimiter_sl', dest='output_attr_delim', 
                                default=args.output_attr_delim)
            parser.add_argument('--output_token_delim', default=constants.SL_TOKEN_DELIM)
            parser.add_argument('--output_empty_line', default=False)

        elif args.output_data_format == 'wl':
            parser.add_argument('--output_data_format', default=args.output_data_format)
            parser.add_argument('--output_attr_delim', default=constants.WL_ATTR_DELIM)
            parser.add_argument('--output_token_delim', default=constants.WL_TOKEN_DELIM)
            parser.add_argument('--output_empty_line', default=True)
            
        elif not args.output_data_format:
            parser.add_argument('--output_data_format', default='wl')
            parser.add_argument('--output_attr_delim', default=constants.WL_ATTR_DELIM)
            parser.add_argument('--output_token_delim', default=constants.WL_TOKEN_DELIM)
            parser.add_argument('--output_empty_line', default=True)

        else:
            print('Error: output data format for task={}/mode={}' .format(args.task, args.execute_mode)
                  + ' must be specified among from {sl, wl}.'
                  + ' Output: {}'.format(args.output_data_format), file=sys.stderr)
            sys.exit()
                

    def add_sgd_options(self, parser, args):
        parser.add_argument('--sgd_lr_decay', default=args.sgd_lr_decay)
        parser.add_argument('--sgd_cyclical_lr', default=args.sgd_cyclical_lr)
        parser.add_argument('--sgd_momentum_ratio', dest='momentum', type=float, default=args.momentum)

        if args.sgd_lr_decay and args.sgd_cyclical_lr:
            print('Error: sgd_lr_decay and sgd_cyclical_lr can not be specified simultaneously',
                  file=sys.stderr)
            sys.exit()


        if args.sgd_cyclical_lr:
            array = args.sgd_cyclical_lr.split(':')
            stepsize = int(array[0])
            min_lr = float(array[1])
            max_lr = float(array[2])
            parser.add_argument('--clr_stepsize', type=int, default=stepsize)
            parser.add_argument('--clr_min_lr', type=float, default=min_lr)
            parser.add_argument('--clr_max_lr', type=float, default=max_lr)
            parser.add_argument('--learning_rate', type=float, default=min_lr)

        else:
            parser.add_argument('--learning_rate', type=float, default=args.learning_rate)

            if args.sgd_lr_decay:
                array = args.sgd_lr_decay.split(':')
                start = int(array[0])
                width = int(array[1])
                rate = float(array[2])
                parser.add_argument('--sgd_lr_decay_start', type=int, default=start)
                parser.add_argument('--sgd_lr_decay_width', type=int, default=width)
                parser.add_argument('--sgd_lr_decay_rate', type=float, default=rate)


    def add_adam_options(self, parser, args):
        parser.add_argument('--adam_alpha', type=float, default=args.adam_alpha)
        parser.add_argument('--adam_beta1', type=float, default=args.adam_beta1)
        parser.add_argument('--adam_beta2', type=float, default=args.adam_beta2)

        
    def add_adadelta_options(self, parser, args):
        parser.add_argument('--adadelta_rho', type=float, default=args.adadelta_rho)


    def add_adagrad_options(self, parser, args):
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)


    def add_rmsprop_options(self, parser, args):
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)
        parser.add_argument('--rmsprop_alpha', type=float, default=args.rmsprop_alpha)


    def add_task_dependent_options(self, parser, args):
        train = args.execute_mode == 'train'
        if train:
            # parser.add_argument('--dump_train_data', default=args.dump_train_data)
            pass

        if (common.is_tagging_task(args.task) or
            common.is_parsing_task(args.task) or
            common.is_attribute_annotation_task(args.task)):
            if train:
                parser.add_argument('--subpos_depth', type=int, default=args.subpos_depth)
        else:
            if train:
                parser.add_argument('--subpos_depth', type=int, default=-1)

        if common.is_attribute_annotation_task(args.task):
            pass

        elif common.is_dual_st_task(args.task):
            parser.add_argument('--unigram_embed_model_path', default=args.unigram_embed_model_path)

            if train:
                parser.add_argument('--submodel_path', default=args.submodel_path)
                parser.add_argument('--submodel_usage', default=args.submodel_usage)

            parser.add_argument('--rnn_dropout', type=float, default=args.rnn_dropout)
            parser.add_argument('--mlp_dropout', type=float, default=args.mlp_dropout)

        else:
            parser.add_argument('--unigram_embed_model_path', default=args.unigram_embed_model_path)

            if train:
                parser.add_argument('--fix_pretrained_embed', action='store_true')
                parser.add_argument('--pretrained_embed_usage', default=args.pretrained_embed_usage)
                if (not args.model_path and args.unigram_embed_model_path and
                    not (args.pretrained_embed_usage == 'init' or args.pretrained_embed_usage == 'concat' or
                         args.pretrained_embed_usage == 'add')):
                    print('Error: pretrained_embed_usage must be specified among from {init, concat, add} '
                          + ':{}'.format(args.pretrained_embed_usage), file=sys.stderr)
                    sys.exit()

                parser.add_argument('--unigram_embed_dim', type=int, default=args.unigram_embed_dim)
                parser.add_argument('--subtoken_embed_dim', type=int, default=args.subtoken_embed_dim)
                parser.add_argument('--rnn_unit_type', default=args.rnn_unit_type)
                parser.add_argument('--rnn_bidirection', action='store_true', default=args.rnn_bidirection)
                parser.add_argument('--rnn_n_layers', type=int, default=args.rnn_n_layers)
                parser.add_argument('--rnn_n_units', type=int, default=args.rnn_n_units)

            parser.add_argument('--embed_dropout', type=float, default=args.embed_dropout)
            parser.add_argument('--rnn_dropout', type=float, default=args.rnn_dropout)

            if common.is_single_st_task(args.task):
                parser.add_argument('--bigram_freq_threshold', type=int, default=args.bigram_freq_threshold)
                parser.add_argument('--bigram_max_vocab_size', type=int, default=args.bigram_max_vocab_size)
                parser.add_argument('--bigram_embed_dim', type=int, default=args.bigram_embed_dim)
                parser.add_argument('--bigram_embed_model_path', default=args.bigram_embed_model_path)
                parser.add_argument('--tokentype_embed_dim', type=int, default=args.tokentype_embed_dim)
                parser.add_argument('--mlp_dropout', type=float, default=args.mlp_dropout)
                parser.add_argument('--mlp_n_layers', type=int, default=args.mlp_n_layers)
                parser.add_argument('--mlp_n_units', type=int, default=args.mlp_n_units)
                parser.add_argument('--inference_layer_type', dest='inference_layer', 
                                         default=args.inference_layer)

                if (not args.model_path and args.bigram_embed_model_path):
                    if args.bigram_embed_dim <= 0:
                        print('Error: bigram_embed_dim must be positive value to use '
                              + 'pretrained bigram embed model: {}'.format(args.bigram_embed_dim),
                              file=sys.stderr)
                        sys.exit()

                    if not (args.pretrained_embed_usage == 'init' or args.pretrained_embed_usage == 'concat' or
                            args.pretrained_embed_usage == 'add'):
                        print('Error: pretrained_embed_usage must be specified among from {init, concat, add} '
                          + ':{}'.format(args.pretrained_embed_usage), file=sys.stderr)
                        sys.exit()

            if common.is_segmentation_task(args.task):
                parser.add_argument('--biaffine_type', default=args.biaffine_type)
                parser.add_argument('--chunk_loss_ratio', type=float, default=args.chunk_loss_ratio)
                parser.add_argument('--chunk_freq_threshold', type=int, default=args.chunk_freq_threshold)
                parser.add_argument('--chunk_max_vocab_size', type=int, default=args.chunk_max_vocab_size)
                parser.add_argument('--rnn_n_layers2', type=int, default=args.rnn_n_layers2)
                parser.add_argument('--rnn_n_units2', type=int, default=args.rnn_n_units2)
                parser.add_argument('--chunk_embed_model_path', default=args.chunk_embed_model_path)
                parser.add_argument('--chunk_pooling_type', default=args.chunk_pooling_type)
                parser.add_argument('--max_chunk_len', type=int, default=args.max_chunk_len)
                parser.add_argument('--biaffine_dropout', type=float, default=args.biaffine_dropout)
                parser.add_argument('--chunk_vector_dropout', type=float, default=args.chunk_vector_dropout)
                parser.add_argument('--chunk_vector_element_dropout', 
                                    type=float, default=args.chunk_vector_element_dropout)
                parser.add_argument('--use_gold_chunk', action='store_true')
                parser.add_argument('--ignore_unknown_pretrained_chunk', action='store_true')

                if train:
                    if args.task == constants.TASK_HSEG:
                        parser.add_argument('--external_dic_path', default='')
                        parser.add_argument('--feature_template', default='')
                        parser.add_argument('--chunk_embed_dim', type=int, default=args.chunk_embed_dim)

                    else:
                        parser.add_argument('--external_dic_path', default=args.external_dic_path)
                        parser.add_argument('--feature_template', default=args.feature_template)
                        if args.external_dic_path and not args.feature_template:
                            print('Warning: loaded external dictionary is not used '
                                  + 'unless specify feature_template', file=sys.stderr)

            else:
                if common.is_tagging_task(args.task):
                    if args.external_dic_path:
                        print('Error: external dictionary can not be used for task={}'.format(args.task), 
                              file=sys.stderr)
                        sys.exit()
                        
                    if args.feature_template:
                        print('Error: feature template can not be used for task={}'.format(args.task), 
                              file=sys.stderr)
                        sys.exit()

                    parser.add_argument('--external_dic_path', default='')
                    parser.add_argument('--feature_template', default='')
                

            if common.is_parsing_task(args.task):
                parser.add_argument('--pos_embed_dim', type=int, default=args.pos_embed_dim)
                parser.add_argument('--mlp4arcrep_n_layers', type=int,   default=args.mlp4arcrep_n_layers)
                parser.add_argument('--mlp4arcrep_n_units', type=int,    default=args.mlp4arcrep_n_units)
                parser.add_argument('--mlp4labelrep_n_layers', type=int, default=args.mlp4labelrep_n_layers)
                parser.add_argument('--mlp4labelrep_n_units', type=int,  default=args.mlp4labelrep_n_units)
                parser.add_argument('--mlp4labelpred_n_layers', type=int, default=args.mlp4labelpred_n_layers)
                parser.add_argument('--mlp4labelpred_n_units', type=int, default=args.mlp4labelpred_n_units)
                parser.add_argument('--hidden_mlp_dropout', type=float,  default=args.hidden_mlp_dropout) 
                parser.add_argument('--pred_layers_dropout', type=float, default=args.pred_layers_dropout) 
