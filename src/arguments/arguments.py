import argparse
import sys

import constants


class ArgumentLoader(object):
    def __init__(self):
        # self.parser = argparse.ArgumentParser()
        pass


    def parse_args(self):
        all_args = self.get_full_parser().parse_args()
        min_args = self.get_minimum_parser(all_args).parse_args()
        return min_args


    def get_full_parser(self):
        parser = argparse.ArgumentParser()

        ### mode options
        parser.add_argument('--execute_mode', '-x', help=
                            'Choose a mode from among \'train\', \'eval\', \'decode\' and \'interactive\'')
        parser.add_argument('--task', '-t', help='Choose a task')
        parser.add_argument('--quiet', '-q', action='store_true',
                            help='Do not output log file and serialized model file')

        ### gpu options
        parser.add_argument('--gpu', '-g', type=int, default=-1, help=
                            'GPU device ID (Use CPU if specify a negative value)')
        parser.add_argument('--cudnn', '-c', dest='use_cudnn', action='store_true', help='Use cuDNN')

        ### training parameters
        parser.add_argument('--epoch_begin', type=int, default=1, help=
                            'Conduct training from i-th epoch (Default: 1)')
        parser.add_argument('--epoch_end', '-e', type=int, default=10, help=
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
        parser.add_argument('--sgd_cyclical_lr', default='', help='Use cyclical learning rate in SGD '
                            + 'in form of \'stepsize:min_lr:max_lr\'')
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
                            'npz/pkl file path of the trained model. \'xxx.hyp\' and \'xxx.s2i\' files '
                            + 'are also read simultaneously when you specify \'xxx_izzz.npz/pkl\' file')

        parser.add_argument('--input_data_path_prefix', '-p', dest='path_prefix', default='', 
                            help='Path prefix of input data')
        parser.add_argument('--train_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of training data')
        parser.add_argument('--devel_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of development data')
        parser.add_argument('--test_data', default='', help=
                            'File path succeeding \'input_data_path_prefix\' of test data')
        parser.add_argument('--decode_data', default='', help=
                            'File path of input text which succeeds \'input_data_path_prefix\'')
        parser.add_argument('--output_data', default='', help=
                            'File path to output parsed text')
        parser.add_argument('--input_data_format', '-f', default='', help=
                            'Choose format of input data among from \'wl\' and \'sl\'')
        parser.add_argument('--output_data_format', default='', help=
                            'Choose format of output data among from \'wl\' and \'sl\'')
        parser.add_argument('--output_attribute_delimiter_sl', dest='output_attr_delim', 
                            default=constants.SL_ATTR_DELIM, help=
                            'Specify delimiter symbol between attributes (word, pos etc.) for SL format '
                            + 'when output analysis results on decode/interactive mode (Default \'/\')')

        ### options for data pre/post-processing
        parser.add_argument('--token_column_index', type=int, dest='token_index', default=0, help=
                            'Index of token column in input data (Default: 0)')
        parser.add_argument('--lowercase_alphabets',  dest='lowercasing', action='store_true', help=
                            'Lowercase alphabets in input text')
        parser.add_argument('--normalize_digits',  action='store_true', help=
                            'Normalize digits by the same symbol in input text')
        parser.add_argument('--ignored_labels', default='', help=''
                            'Sepecify labels to be ignored on evaluation '
                            + 'by format \'label_1,label_2,...,label_N\'')

        ### model parameters
        parser.add_argument('--token_freq_threshold', type=int, default=1, help=
                            'Token frequency threshold. Tokens whose frequency are lower than'
                            + 'the the threshold are regarded as unknown tokens (Default: 1)')
        parser.add_argument('--token_max_vocab_size', type=int, default=-1, help=
                            'Maximum size of token vocaburaly. '
                            + 'low frequency tokens are regarded as unknown tokens so that '
                            + 'vocaburaly size does not exceed the specified size so much '
                            + 'if set positive value (Default: -1)')

        ### neural models
        parser.add_argument('--pretrained_embed_usage', default='none', help=
                            'Specify usage of pretrained embedding model among from '
                            + '{init, concat, add}')

        return parser


    def get_minimum_parser(self, args):
        parser = argparse.ArgumentParser()

        # basic options
        self.add_basic_options(parser, args)

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


    def add_basic_options(self, parser, args):
        # mode options
        parser.add_argument('--execute_mode', '-x', required=True, default=args.execute_mode)
        if args.model_path:
            parser.add_argument('--task', '-t', default=args.task)
        else:
            parser.add_argument('--task', '-t', required=True, default=args.task)
        parser.add_argument('--quiet', '-q', action='store_true', default=args.quiet)

        # options for data pre/post-processing
        parser.add_argument('--token_column_index', type=int, dest='token_index', default=args.token_index)
        parser.add_argument('--lowercase_alphabets',  dest='lowercasing', action='store_true',
                                 default=args.lowercasing)
        parser.add_argument('--normalize_digits',  action='store_true', default=args.normalize_digits)

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
        parser.add_argument('--pretrained_embed_usage', default=args.pretrained_embed_usage)

        # optimizer parameters
        parser.add_argument('--optimizer', '-o', default=args.optimizer)
        parser.add_argument('--grad_clip', type=float, default=args.grad_clip)
        parser.add_argument('--weight_decay_ratio', type=float, default=args.weight_decay_ratio)

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
        parser.add_argument('--train_data', default=args.train_data)
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
        parser.add_argument('--input_data_format', '-f', default=args.input_data_format)
            

    def add_output_data_format_options(self, parser, args):
        if args.output_data_format == 'sl':
            parser.add_argument('--output_data_format', default=args.output_data_format)
            parser.add_argument('--output_attribute_delimiter_sl', dest='output_attr_delim', 
                                default=args.output_attr_delim)
            parser.add_argument('--output_token_delim', default=constants.SL_TOKEN_DELIM)
            # parser.add_argument('--output_empty_line', default=False)

        elif args.output_data_format == 'wl':
            parser.add_argument('--output_data_format', default=args.output_data_format)
            parser.add_argument('--output_attr_delim', default=constants.WL_ATTR_DELIM)
            parser.add_argument('--output_token_delim', default=constants.WL_TOKEN_DELIM)
            # parser.add_argument('--output_empty_line', default=True)
            
        elif not args.output_data_format:
            parser.add_argument('--output_data_format', default='wl')
            parser.add_argument('--output_attr_delim', default=constants.WL_ATTR_DELIM)
            parser.add_argument('--output_token_delim', default=constants.WL_TOKEN_DELIM)
            # parser.add_argument('--output_empty_line', default=True)

        else:
            print('Error: output data format must be specified among from {sl, wl}'
                  + ': {}'.format(args.output_data_format), file=sys.stderr)
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
