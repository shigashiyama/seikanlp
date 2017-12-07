import argparse


class Arguments(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        parser = self.parser

        ### used letters: bcegmopqtuvx

        # mandatory options
        parser.add_argument('--execute_mode', '-x', required=True,
                            help='Choose a mode from among \'train\', \'eval\', \'decode\'' 
                            + 'and \'interactive\'')
        # gpu options
        parser.add_argument('--gpu', '-g', type=int, default=-1, help=
                            'GPU device ID (Use CPU if unspecify any values or specify a negative value)')
        parser.add_argument('--cudnn', dest='use_cudnn', action='store_true',
                            help='Use cuDNN')

        # training parameters
        parser.add_argument('--epoch_begin', type=int, default=1, 
                            help='Conduct training from i-th epoch (Default: 1)')
        parser.add_argument('--epoch_end', '-e', type=int, default=5,
                            help='Conduct training up to i-th epoch (Default: 5)')
        parser.add_argument('--evaluation_size', type=int, default=10000, 
                            help='The number of examples which'
                            + ' trained model is evaluated and saved at each multiple of'
                            + ' (Default: 10000)')
        parser.add_argument('--batchsize', '-b', type=int, default=100,
                            help='The number of examples in each mini-batch (Default: 100)')
        parser.add_argument('--gradclip', '-c', type=float, default=5,
                            help='Gradient norm threshold to clip')

        # optimizer parameters
        parser.add_argument('--optimizer', default='sgd',
                            help='Choose optimizing algorithm from among \'sgd\', \'adam\' and \'adagrad\''
                            + ' (Default: sgd)')
        parser.add_argument('--learning_rate', type=float, default=1.0, 
                            help='Initial learning rate (Default: 1.0)')
        parser.add_argument('--momentum', type=float, default=0.0, help='Momentum ratio for SGD')
        parser.add_argument('--adam_alpha', type=float, default=0.001, help='alpha for Adam')
        parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1 for Adam')
        parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 for Adam')
        parser.add_argument('--lrdecay', default='', 
                            help='Specify information on learning rate decay'
                            + ' by format \'start:width:rate\''
                            + ' where start indicates the number of epochs to start decay,'
                            + ' width indicates the number epochs maintaining the same decayed learning rate,'
                            + ' and rate indicates the decay late to multipy privious learning late')
        parser.add_argument('--weightdecay', type=float, default=0.0, 
                            help='Weight decay ratio (Default: 0.0)')

        # options to resume training
        parser.add_argument('--model_path', '-m',
                            help='npz File path of the trained model. \'xxx.hyp\' and \'xxx.s2i\' files'
                            + 'are also read simultaneously if you specify \'xxx_izzz.npz\' file.')

        # model parameters
        parser.add_argument('--freq_threshold', type=int, default=1,
                            help='The threshold of frequency to regard tokens as unknown tokens'
                            + '(Default: 1)')
        parser.add_argument('--max_vocab_size', type=int, default=-1,
                            help='')
        parser.add_argument('--no_freq_update', action='store_true',
                            help='')
        parser.add_argument('--token_embed_dim', '-d', type=int, default=300,
                            help='The number of dimension of token (character or word) embedding'
                            + '(Default: 300)')

        # data paths and related options
        parser.add_argument('--path_prefix', '-p', help='Path prefix of input data')
        parser.add_argument('--train_data', '-t', default='',
                            help='File path succeeding \'path_prefix\' of training data')
        parser.add_argument('--valid_data', '-v', default='',
                            help='File path succeeding \'path_prefix\' of validation data')
        parser.add_argument('--test_data', default='',
                            help='File path succeeding \'path_prefix\' of test data')
        parser.add_argument('--raw_data', '-r', default='',
                            help='File path of input raw text which succeeds \'path_prefix\'')
        parser.add_argument('--label_reference_data', default='',
                            help='File path succeeding \'path_prefix\''
                            + ' of data with the same format as training data to load pre-defined labels')
        parser.add_argument('--output_path', '-o', default='',
                            help='File path to output parsed text')
        parser.add_argument('--dump_train_data', action='store_true',
                            help='Dump data specified as \'train_data\''
                            + ' using new file name endding with \'.pickle\'')
        parser.add_argument('--token_embed_model_path', 
                            help='File path of pretrained model of token (character or word) embedding')
        parser.add_argument('--fix_pretrained_embed', action='store_true',
                            help='')

        # options for data pre/post-processing
        parser.add_argument('--lowercase',  action='store_true',
                            help='Lowercase alphabets in the case of using English data')
        parser.add_argument('--normalize_digits',  action='store_true',
                            help='Normalize digits by the same symbol in the case of using English data')
        parser.add_argument('--ignore_labels', default=set(),
                            help='')

        # other options
        parser.add_argument('--quiet', '-q', action='store_true',
                            help='Do not output log file and serialized model file')

    def parse_arguments(self):
        args = self.parser.parse_args()
        if args.lrdecay:
            self.parser.add_argument('lrdecay_start')
            self.parser.add_argument('lrdecay_width')
            self.parser.add_argument('lrdecay_rate')
            array = args.lrdecay.split(':')
            args.lrdecay_start = int(array[0])
            args.lrdecay_width = int(array[1])
            args.lrdecay_rate = float(array[2])

        if args.execute_mode == 'interactive':
            args.quiet = True

        return args


class TaggerArguments(Arguments):
    def __init__(self):
        super(TaggerArguments, self).__init__()
        parser = self.parser

        ### used letters: + dflsuw

        # model parameters
        parser.add_argument('--dropout', type=float, default=0.0, 
                            help='Dropout ratio for RNN vertical layers (Default: 0.0)')
        parser.add_argument('--rnn_unit_type', default='lstm',
                            help='Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain\''
                            + ' (Default: lstm)')
        parser.add_argument('--rnn_bidirection', action='store_true', 
                            help='Use bidirectional RNN')
        parser.add_argument('--rnn_n_layers', '-l', type=int, default=1, 
                            help='The number of RNN layers (Default: 1)')
        parser.add_argument('--rnn_n_units', type=int, default=800,
                            help='The number of hidden units of RNN (Default: 800)')
        parser.add_argument('--inference_layer', default='crf',
                            help='Choose type of inference layer from between \'softmax\' and \'crf\''
                            + ' (Default: crf)')

        # data paths and related options
        parser.add_argument('--data_format', '-f', 
                        help='Choose format of input data among from'
                        # + ' \'seg\' (word segmentation on data where words are split with single spaces),'
                        # + ' \'seg_tag\' (word segmentation w/ POS tagging on data '
                        # + ' where \'words_pos\' pairs are split with single spaces),'
                        # + ' \'bccwj_seg\' (word segmentation on data with BCCWJ format),'
                        # + ' \'bccwj_seg_tag\' (word segmentation w/ POS tagging on data with BCCWJ format),'
                        # + ' \'bccwj_tag\' (POS tagging from gold words on data with BCCWJ format),'
                        # + ' \'wsj\' (POS tagging on data with CoNLL-2005 format),'
                        # + ' \'conll2003\' (NER on data with CoNLL-2003 format),'
        )
        parser.add_argument('--subpos_depth', '-s', type=int, default=-1,
                            help='Set positive integer (use POS up to i-th hierarcy)'
                            + ' or other value (use POS with all sub POS) when set \'bccwj_seg_tag\''
                            + ' to data_format')
        parser.add_argument('--dict_path', 
                            help='File path of word dictionary that lists words, or words and POS')
        parser.add_argument('--feature_template', default='',
                            help='Use dictionary features based on given feature template file.'
                            + ' Specify \'defualt\' to use default features defined for each task,'
                            + ' or specify the path of castomized template file.')


class ParserArguments(Arguments):
    def __init__(self):
        super(ParserArguments, self).__init__()
        parser = self.parser

        ### used letters: + fls

        # model parameters
        parser.add_argument('--rnn_dropout', type=float, default=0.0, 
                            help='Dropout ratio for RNN vertical layers (Default: 0.0)')
        parser.add_argument('--hidden_mlp_dropout', type=float, default=0.0, 
                            help='Dropout ratio for MLP (Default: 0.0)')
        parser.add_argument('--pred_layers_dropout', type=float, default=0.0, 
                            help='Dropout ratio for prediction layers (Default: 0.0)')
        parser.add_argument('--pos_embed_dim', type=int, default=100,
                            help='The number of dimension of pos embedding (Default: 100)')
        parser.add_argument('--rnn_unit_type', default='lstm',
                            help='Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain\''
                            + ' (Default: lstm)')
        parser.add_argument('--rnn_bidirection', action='store_true', 
                            help='Use bidirectional RNN')
        parser.add_argument('--rnn_n_layers', '-l', type=int, default=1, 
                            help='The number of RNN layers (Default: 1)')
        parser.add_argument('--rnn_n_units', type=int, default=400,
                            help='The number of hidden units of RNN (Default: 400)')
        parser.add_argument('--mlp4pospred_n_layers', type=int, default=0, 
                            help='')
        parser.add_argument('--mlp4pospred_n_units', type=int, default=200,
                            help='')
        parser.add_argument('--mlp4arcrep_n_layers', type=int, default=1, 
                            help='')
        parser.add_argument('--mlp4arcrep_n_units', type=int, default=200,
                            help='')
        parser.add_argument('--mlp4labelrep_n_layers', type=int, default=1, 
                            help='')
        parser.add_argument('--mlp4labelrep_n_units', type=int, default=200,
                            help='')
        parser.add_argument('--mlp4labelpred_n_layers', type=int, default=1, 
                            help='')
        parser.add_argument('--mlp4labelpred_n_units', type=int, default=200,
                            help='')

        # data paths and related options
        parser.add_argument('--data_format', '-f', 
                        help='Choose format of input data among from'
        )
        parser.add_argument('--subpos_depth', '-s', type=int, default=-1,
                            help='Set positive integer (use POS up to i-th hierarcy)'
                            + ' or other value (use POS with all sub POS) when set \'bccwj_seg_tag\''
                            + ' to data_format')
