import sys

import constants
from arguments.arguments import ArgumentLoader


class TaggerArgumentLoader(ArgumentLoader):
    def parse_args(self):
        return super().parse_args()


    def get_full_parser(self):
        parser = super().get_full_parser()

        ### data paths and related options
        parser.add_argument('--unigram_embed_model_path', default='', help=
                            'File path of pretrained model of token (character or word) unigram embedding')
        parser.add_argument('--bigram_embed_model_path', default='', help=
                            'File path of pretrained model of token (character or word) bigram embedding')
        parser.add_argument('--external_dic_path', help=
                            'File path of external word dictionary listing known words')
        parser.add_argument('--feature_template', default='', help=
                            'Use dictionary features based on given feature template file. '
                            + 'Specify \'defualt\' to use default features '
                            + 'or specify the path of castomized template file')
        parser.add_argument('--batch_feature_extraction', action='store_true', help=
                            'If True, extract dictionary features for each mini-batch '
                            + 'to save memory. Otherwise, extract features just one time '
                            'before starting training')
        ### options for data pre/post-processing
        # common
        parser.add_argument('--attribute_column_indexes', dest='attr_indexes', default='', help=
                            'Indices of attribute columns in input data '
                            + '(Multiple values can be specified such as \'1,2,3\' '
                            + 'but only the first value is used and the others are ignored)')
        parser.add_argument('--attribute_depths', dest='attr_depths', default='', help=
                            'Set positive integer (use attribute up to i-th hierarcy), or '
                            + '0 or negative value (use all sub-attribute). '
                            + 'atribute hierarchy must be indicated by \'-\' like \'NOUN-GENERAL\'')
        parser.add_argument('--attribute_chunking_flags', dest='attr_chunking_flags', default='', help=
                            'Set \'True\' if use chunking labels like \'B-X\', \'I-X\' '
                            + 'instead of original attribute like X ')
        parser.add_argument('--attribute_target_labelsets', dest='attr_target_labelsets', default='', help=
                            'Sepecify labels to be considered '
                            + 'to ignore the rest unspecified labels'
                            + 'by format \'label_1,label_2,...,label_N\'')
        parser.add_argument('--attribute_delimiter', dest='attr_delim', default='', help=
                            'Specify delimiter symbol of input data with sl format')
        # sequence labeling
        parser.add_argument('--tagging_unit', default='single', help=
                            'Specify \'single\' for character-based model or'
                            + '\'hybrid\' for hybrid model of character and word.')

        ### model parameters

        ### options for model architecture and parameters
        # common
        parser.add_argument('--embed_dropout', type=float, default=0.0, help=
                            'Dropout ratio for embedding layers (Default: 0.0)')
        parser.add_argument('--rnn_dropout', type=float, default=0.0, help=
                            'Dropout ratio for RNN vertical layers (Default: 0.0)')
        parser.add_argument('--rnn_unit_type', default='lstm', help=
                            'Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain\' (Default: lstm)')
        parser.add_argument('--rnn_bidirection', action='store_true', help='Use bidirectional RNN')
        parser.add_argument('--rnn_n_layers', type=int, default=1, help=
                            'The number of RNN layers (Default: 1)')
        parser.add_argument('--rnn_n_units', type=int, default=600, help=
                            'The number of hidden units of RNN (Default: 600)')
        # segmentation
        parser.add_argument('--evaluation_method', default='normal', help='')
        # segmentation/tagging
        parser.add_argument('--bigram_freq_threshold', type=int, default=1, help=
                            'Token bigram frequency threshold. Bigrams whose frequency are lower than'
                            + 'the the threshold are regarded as unknown bigrams (Default: 1)')
        parser.add_argument('--bigram_max_vocab_size', type=int, default=-1, help=
                            'Maximum size of token bigram vocaburaly. '
                            + 'low frequency bigrams  are regarded as unknown bigrams so that '
                            + 'vocaburaly size does not exceed the specified size so much '
                            + 'if set positive value (Default: -1)')
        parser.add_argument('--mlp_dropout', type=float, default=0.0, help=
                            'Dropout ratio for MLP of sequence labeling model (Default: 0.0)')
        parser.add_argument('--unigram_embed_dim', type=int, default=300, help=
                            'The number of dimension of token (character or word) unigram embedding'
                            + '(Default: 300)')
        parser.add_argument('--bigram_embed_dim', type=int, default=0, help=
                            'The number of dimension of token bigram embedding (Default: 0)')
        parser.add_argument('--attr1_embed_dim', type=int, default=0, help=
                            'The number of dimension of 1st attibute embedding (Default: 0)')
        parser.add_argument('--mlp_n_layers', type=int, default=1, help=
                            'The number of layers of MLP of sequence labeling model. '
                            + 'The last layer projects input hidden vector to '
                            + 'dimensional space of number of labels (Default: 1)')
        parser.add_argument('--mlp_n_units', type=int, default=300, help=
                            'The number of hidden units of MLP of sequence labeling model (Default: 300)')
        parser.add_argument('--inference_layer_type', dest='inference_layer', default='crf', help=
                            'Choose type of inference layer for sequence labeling model from between '
                            + '\'softmax\' and \'crf\' (Default: crf)')
        # hybrid unit segmentation/tagging
        parser.add_argument('--biaffine_type', default='', help=
                            'Specify \'u\', \'v\', and/or \'b\' if use non-zero matrices/vectors '
                            + 'in biaffine transformation \'x1*W*x2+x1*U+x2*V+b\' such as \'uv\'')
        parser.add_argument('--chunk_freq_threshold', type=int, default=1, help=
                            'Chunk frequency threshold. Chunks whose frequency are lower than'
                            + 'the the threshold are regarded as unknown chunks (Default: 1)')
        parser.add_argument('--chunk_max_vocab_size', type=int, default=-1, help=
                            'Maximum size of chunk vocaburaly. '
                            + 'low frequency chunks are regarded as unknown chunks so that '
                            + 'vocaburaly size does not exceed the specified size so much '
                            + 'if set positive value (Default: -1)')
        parser.add_argument('--rnn_n_layers2', type=int, default=0, help=
                            'The number of 2nd RNN layers (Default: 0)')
        parser.add_argument('--rnn_n_units2', type=int, default=0, help=
                            'The number of hidden units of 2nd RNN (Default: 0)')
        parser.add_argument('--chunk_embed_dim', type=int, default=300, help=
                            'The number of dimension of chunk embedding (Default: 0)')
        parser.add_argument('--chunk_embed_model_path', default='', help=
                            'File path of pretrained model of chunk (typically word) embedding')
        parser.add_argument('--chunk_pooling_type', default=constants.WAVG, help=
                            'Specify integration method of chunk vector from among '
                            + '{avg, wavg, con, wcon}')
        parser.add_argument('--min_chunk_len', type=int, default=1)
        parser.add_argument('--max_chunk_len', type=int, default=4)
        parser.add_argument('--use_gold_chunk', action='store_true') # always True
        parser.add_argument('--biaffine_dropout', type=float, default=0.0, help=
                            'Dropout ratio for biaffine layer in hybrid model (Default: 0.0)')
        parser.add_argument('--chunk_vector_dropout', type=float, default=0.0, help=
                            'Dropout ratio for chunk vectors in hybrid model (Default: 0.0)')
        parser.add_argument('--unuse_nongold_chunk', action='store_const', const=False) # always False
        parser.add_argument('--ignore_unknown_pretrained_chunk', action='store_const', const=False) # always False
        parser.add_argument('--gen_oov_chunk_for_test', action='store_const', const=False) # always False

        return parser
        

    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)
        parser.add_argument('--evaluation_method', default=args.evaluation_method)
        if not (args.evaluation_method == 'normal' or
                args.evaluation_method == 'each_length' or 
                args.evaluation_method == 'each_vocab' or
                args.evaluation_method == 'attention' or
                args.evaluation_method == 'stat_test'
        ):
            print('Error: evaluation_method must be specified among from '
                  + '{normal, each_length, each_vocab, attention}:{}'.format(
                      args.evaluation_method), file=sys.stderr)
            sys.exit()

        parser.add_argument('--attribute_column_indexes', dest='attr_indexes', default=args.attr_indexes)
        parser.add_argument('--attribute_depths', dest='attr_depths', default=args.attr_depths)
        parser.add_argument('--attribute_chunking_flags', dest='attr_chunking_flags',
                            default=args.attr_chunking_flags)
        parser.add_argument('--attribute_target_labelsets', dest='attr_target_labelsets',
                            default=args.attr_target_labelsets)
        parser.add_argument('--attribute_delimiter', dest='attr_delim', default=args.attr_delim)
        # parser.add_argument('--attribute_predictions', dest='attr_predictions', 
        #                     default=args.attribute_predictions)

        if (args.task == constants.TASK_TAG or args.task == constants.TASK_SEGTAG):
            if args.execute_mode == 'train' and not args.attr_indexes:
                print('Error: attribute_column_indexes must be specified for task='
                      + '{}'.format(args.task), file=sys.stderr)
                sys.exit()

        parser.add_argument('--unigram_embed_model_path', default=args.unigram_embed_model_path)
        parser.add_argument('--embed_dropout', type=float, default=args.embed_dropout)
        parser.add_argument('--rnn_dropout', type=float, default=args.rnn_dropout)

        # specific options for segmentation/tagging
        parser.add_argument('--bigram_embed_model_path', default=args.bigram_embed_model_path)
        parser.add_argument('--external_dic_path', default=args.external_dic_path)
        parser.add_argument('--feature_template', default=args.feature_template)
        parser.add_argument('--batch_feature_extraction', action='store_true', 
                            default=args.batch_feature_extraction)
        parser.add_argument('--mlp_dropout', type=float, default=args.mlp_dropout)
        parser.add_argument('--tagging_unit', default=args.tagging_unit)
        # if args.external_dic_path and not args.feature_template:
        #     print('Warning: loaded external dictionary is not used '
        #           + 'unless feature_template is specified.', file=sys.stderr)

        # specific options for hybrid unit segmentation/tagging
        if args.tagging_unit == 'hybrid':
            parser.add_argument('--biaffine_dropout', type=float, default=args.biaffine_dropout)
            parser.add_argument('--chunk_vector_dropout', type=float, default=args.chunk_vector_dropout)
            parser.add_argument('--chunk_embed_model_path', default=args.chunk_embed_model_path)
            parser.add_argument('--gen_oov_chunk_for_test', action='store_const', const=False, default=False)

        if args.execute_mode == 'train':
            if (not args.model_path and 
                (args.unigram_embed_model_path or args.bigram_embed_model_path)):
                if not (args.pretrained_embed_usage == 'init' or 
                        args.pretrained_embed_usage == 'concat' or
                        args.pretrained_embed_usage == 'add'):
                    print('Error: pretrained_embed_usage must be specified among from {init, concat, add} '
                          + ':{}'.format(args.pretrained_embed_usage), file=sys.stderr)
                    sys.exit()

                if args.unigram_embed_model_path and args.unigram_embed_dim <= 0:
                    print('Error: unigram_embed_dim must be positive value to use '
                          + 'pretrained unigram embed model: {}'.format(args.unigram_embed_dim),
                          file=sys.stderr)
                    sys.exit()

                if args.bigram_embed_model_path and args.bigram_embed_dim <= 0:
                    print('Error: bigram_embed_dim must be positive value to use '
                          + 'pretrained bigram embed model: {}'.format(args.bigram_embed_dim),
                          file=sys.stderr)
                    sys.exit()

            parser.add_argument('--unigram_embed_dim', type=int, default=args.unigram_embed_dim)
            parser.add_argument('--rnn_unit_type', default=args.rnn_unit_type)
            parser.add_argument('--rnn_bidirection', action='store_true', default=args.rnn_bidirection)
            parser.add_argument('--rnn_n_layers', type=int, default=args.rnn_n_layers)
            parser.add_argument('--rnn_n_units', type=int, default=args.rnn_n_units)
            # specific options for segmentation/tagging
            parser.add_argument('--bigram_freq_threshold', type=int, default=args.bigram_freq_threshold)
            parser.add_argument('--bigram_max_vocab_size', type=int, default=args.bigram_max_vocab_size)
            parser.add_argument('--bigram_embed_dim', type=int, default=args.bigram_embed_dim)
            parser.add_argument('--attr1_embed_dim', type=int, default=args.attr1_embed_dim)
            parser.add_argument('--mlp_n_layers', type=int, default=args.mlp_n_layers)
            parser.add_argument('--mlp_n_units', type=int, default=args.mlp_n_units)
            parser.add_argument('--inference_layer_type', dest='inference_layer', 
                                default=args.inference_layer)
            # specific options for hybrid unit segmentation/tagging
            if args.tagging_unit == 'hybrid':
                parser.add_argument('--biaffine_type', default=args.biaffine_type)
                parser.add_argument('--chunk_embed_dim', type=int, default=args.chunk_embed_dim)
                parser.add_argument('--chunk_freq_threshold', type=int, default=args.chunk_freq_threshold)
                parser.add_argument('--chunk_max_vocab_size', type=int, default=args.chunk_max_vocab_size)
                parser.add_argument('--rnn_n_layers2', type=int, default=args.rnn_n_layers2)
                parser.add_argument('--rnn_n_units2', type=int, default=args.rnn_n_units2)
                chunk_pooling_type = args.chunk_pooling_type.upper()
                if (chunk_pooling_type == constants.AVG or
                    chunk_pooling_type == constants.WAVG or
                    chunk_pooling_type == constants.CON or
                    chunk_pooling_type == constants.WCON):
                    parser.add_argument('--chunk_pooling_type', default=chunk_pooling_type)
                else:
                    print('Error: chunk pooling type must be specified among from '
                          + '{AVG, WAVG, CON, WCON}.'
                          + ' Input: {}'.format(args.chunk_pooling_type), file=sys.stderr)
                    sys.exit()
                
                parser.add_argument('--min_chunk_len', type=int, default=args.min_chunk_len, help=
                                    'min length of chunk used in hybrid segmentation model')
                parser.add_argument('--max_chunk_len', type=int, default=args.max_chunk_len, help=
                                    'max length of chunk used in hybrid segmentation model')
                parser.add_argument('--use_gold_chunk', action='store_const', const=True, default=True)
                parser.add_argument('--unuse_nongold_chunk', action='store_const', const=False, default=False)
                parser.add_argument('--ignore_unknown_pretrained_chunk', action='store_const', 
                                    const=False, default=False)

        return parser


    def add_input_data_format_option(self, parser, args):
        if args.execute_mode == 'interactive':
            pass

        elif args.task == constants.TASK_SEG or args.task == constants.TASK_SEGTAG:
            if args.input_data_format == 'sl':
                parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

            elif args.input_data_format == 'wl':
                if args.execute_mode == 'train' or args.execute_mode == 'eval':
                    parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

                elif args.execute_mode == 'decode':
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified as \'sl\'.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()

            else:
                if args.execute_mode == 'decode':
                    parser.add_argument('--input_data_format', '-f', default='sl')

                else:
                    print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                          + ' must be specified among from {sl, wl}.'
                          + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                    sys.exit()

        elif args.task == constants.TASK_TAG:
            if args.input_data_format == 'sl' or args.input_data_format == 'wl':
                parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

            else:
                print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                      + ' must be specified among from {sl, wl}.'
                      + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                sys.exit()

