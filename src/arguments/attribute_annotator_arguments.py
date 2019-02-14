import sys

from arguments.arguments import ArgumentLoader


class AttributeAnnotatorArgumentLoader(ArgumentLoader):
    def get_full_parser(self):
        parser = super().get_full_parser()

        ### options for data pre/post-processing
        parser.add_argument('--label_column_index', type=int, dest='label_index', default=2, help=
                            'Index of label column in input data (Default: 2)')
        parser.add_argument('--attribute_column_indexes', dest='attr_indexes', default='', help='')
        parser.add_argument('--attribute_depths', dest='attr_depths', default='', help=
                            'Set positive integer (use attribute up to i-th hierarcy), or '
                            + '0 or negative value (use all sub-attribute). '
                            + 'atribute hierarchy must be indicated by \'-\' like \'NOUN-GENERAL\'')
        parser.add_argument('--attribute_chunking_flags', dest='attr_chunking_flags', default='', help='')
        parser.add_argument('--attribute_target_labelsets', dest='attr_target_labelsets', default='', help=''
                            'Sepecify labels to be considered'
                            + 'by format \'label_1,label_2,...,label_N\'')
        parser.add_argument('--attribute_delimiter', dest='attr_delim', default='', help='')

        return parser

    
    def add_basic_options(self, parser, args):
        # mode options
        parser.add_argument('--execute_mode', '-x', required=True, default=args.execute_mode)
        parser.add_argument('--task', '-t', default='sematt')
        parser.add_argument('--quiet', '-q', action='store_true', default=args.quiet)

        # options for data pre/post-processing
        parser.add_argument('--token_column_index', type=int, dest='token_index', default=args.token_index)
        parser.add_argument('--lowercase_alphabets',  dest='lowercasing', action='store_true',
                                 default=args.lowercasing)
        parser.add_argument('--normalize_digits',  action='store_true', default=args.normalize_digits)


    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)

        parser.add_argument('--label_column_index', type=int, dest='label_index', default=args.label_index)
        parser.add_argument('--attribute_column_indexes', dest='attr_indexes', default=args.attr_indexes)
        parser.add_argument('--attribute_depths', dest='attr_depths', default=args.attr_depths)
        parser.add_argument('--attribute_chunking_flags', dest='attr_chunking_flags',
                            default=args.attr_chunking_flags)
        parser.add_argument('--attribute_target_labelsets', dest='attr_target_labelsets',
                            default=args.attr_target_labelsets)
        parser.add_argument('--attribute_delimiter', dest='attr_delim', default=args.attr_delim)

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
            parser.add_argument('--input_data_format', '-f', default=args.input_data_format)
