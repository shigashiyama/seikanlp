import sys

from arguments.arguments import ArgumentLoader


class ParserArgumentLoader(ArgumentLoader):
    def get_full_parser(self):
        parser = super().get_full_parser()

        ### data paths and related options
        parser.add_argument('--unigram_embed_model_path', default='', help=
                            'File path of pretrained model of token (character or word) unigram embedding')

        ### options for data pre/post-processing
        parser.add_argument('--head_column_index', type=int, dest='head_index', default=2, help=
                            'Index of head column in input data (Default: 2)')
        parser.add_argument('--arc_column_index', type=int, dest='arc_index', default=3, help=
                            'Index of arc column in input data (Default: 3)')
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

        ### options for model architecture
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

        # parsing
        parser.add_argument('--unigram_embed_dim', type=int, default=300, help=
                            'The number of dimension of token (character or word) unigram embedding'
                            + '(Default: 300)')
        parser.add_argument('--attr1_embed_dim', type=int, default=0, help=
                            'The number of dimension of 1st attribute embedding (Default: 0)')
        parser.add_argument('--hidden_mlp_dropout', type=float, default=0.0, help=
                            'Dropout ratio for MLP of parsing model (Default: 0.0)')
        parser.add_argument('--pred_layers_dropout', type=float, default=0.0, help=
                            'Dropout ratio for prediction layers of parsing model (Default: 0.0)')
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

        return parser

    
    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)

        parser.add_argument('--head_column_index', type=int, dest='head_index', default=args.head_index, help='')
        parser.add_argument('--arc_column_index', type=int, dest='arc_index', default=args.arc_index, help='')
        parser.add_argument('--attribute_column_indexes', dest='attr_indexes', default=args.attr_indexes)
        parser.add_argument('--attribute_depths', dest='attr_depths', default=args.attr_depths)
        parser.add_argument('--attribute_chunking_flags', dest='attr_chunking_flags',
                            default=args.attr_chunking_flags)
        parser.add_argument('--attribute_target_labelsets', dest='attr_target_labelsets',
                            default=args.attr_target_labelsets)
        parser.add_argument('--attribute_delimiter', dest='attr_delim', default=args.attr_delim)

        parser.add_argument('--unigram_embed_model_path', default=args.unigram_embed_model_path)
        parser.add_argument('--embed_dropout', type=float, default=args.embed_dropout)
        parser.add_argument('--rnn_dropout', type=float, default=args.rnn_dropout)
        # specific options for parsing 
        parser.add_argument('--hidden_mlp_dropout', type=float,  default=args.hidden_mlp_dropout) 
        parser.add_argument('--pred_layers_dropout', type=float, default=args.pred_layers_dropout) 
    
        if args.execute_mode == 'train':
            if (not args.model_path and args.unigram_embed_model_path):
                if not (args.pretrained_embed_usage == 'init' or 
                        args.pretrained_embed_usage == 'concat' or
                        args.pretrained_embed_usage == 'add'):
                    print('Error: pretrained_embed_usage must be specified among from {init, concat, add} '
                          + ':{}'.format(args.pretrained_embed_usage), file=sys.stderr)
                    sys.exit()

                if args.unigram_embed_dim <= 0:
                    print('Error: unigram_embed_dim must be positive value to use '
                          + 'pretrained unigram embed model: {}'.format(args.unigram_embed_dim),
                          file=sys.stderr)
                    sys.exit()

            parser.add_argument('--unigram_embed_dim', type=int, default=args.unigram_embed_dim)
            parser.add_argument('--rnn_unit_type', default=args.rnn_unit_type)
            parser.add_argument('--rnn_bidirection', action='store_true', default=args.rnn_bidirection)
            parser.add_argument('--rnn_n_layers', type=int, default=args.rnn_n_layers)
            parser.add_argument('--rnn_n_units', type=int, default=args.rnn_n_units)
            # specific options for parsing
            parser.add_argument('--attr1_embed_dim', type=int, default=args.attr1_embed_dim)
            parser.add_argument('--mlp4arcrep_n_layers', type=int,   default=args.mlp4arcrep_n_layers)
            parser.add_argument('--mlp4arcrep_n_units', type=int,    default=args.mlp4arcrep_n_units)
            parser.add_argument('--mlp4labelrep_n_layers', type=int, default=args.mlp4labelrep_n_layers)
            parser.add_argument('--mlp4labelrep_n_units', type=int,  default=args.mlp4labelrep_n_units)
            parser.add_argument('--mlp4labelpred_n_layers', type=int, default=args.mlp4labelpred_n_layers)
            parser.add_argument('--mlp4labelpred_n_units', type=int, default=args.mlp4labelpred_n_units)

        return parser


    def add_input_data_format_option(self, parser, args):
        if args.execute_mode == 'interactive':
            pass

        elif args.input_data_format == 'wl':
            parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

        elif args.input_data_format == 'sl':
            if args.execute_mode == 'train' or args.execute_mode == 'eval':                
                print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                      + ' must be specified as \'wl\'.'
                      + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                sys.exit()

            else:
                parser.add_argument('--input_data_format', '-f', default=args.input_data_format)

        else:
            if args.execute_mode == 'train' or args.execute_mode == 'eval':                
                parser.add_argument('--input_data_format', '-f', default='wl')

            else:
                print('Error: input data format for task={}/mode={}' .format(args.task, args.execute_mode)
                      + ' must be specified among from {sl, wl}.'
                      + ' Input: {}'.format(args.input_data_format), file=sys.stderr)
                sys.exit()
