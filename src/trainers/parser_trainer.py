import sys

from chainer import cuda

import common
import util
import constants
import data_io
import classifiers
import optimizers
from trainers import trainer
from trainers.trainer import Trainer


class ParserTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.label_begin_index = 3
        # elif args.task == 'tag_dep' or args.task == 'tag_tdep':
        #     self.label_begin_index = 2


    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(self.args.unigram_embed_model_path)


    def init_model(self):
        super().init_model()
        if self.unigram_embed_model:
            finetuning = not self.hparams['fix_pretrained_embed']
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, finetuning=finetuning)


    def load_model(self):
        super().load_model()
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
        if 'pred_layers_dropout' in self.hparams:
            self.classifier.change_pred_layers_dropout_ratio(self.hparams['pred_layers_dropout'])
            

    def update_model(self):
        id2uni = self.dic.tables[constants.UNIGRAM].id2str
        model = self.classifier.predictor

        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            self.classifier.grow_embedding_layers(
                self.dic, self.unigram_embed_model, train=(self.args.execute_mode=='train'))
            self.classifier.grow_inference_layers(self.dic)


    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'fix_pretrained_embed' : self.args.fix_pretrained_embed,
            'unigram_embed_dim' : self.args.unigram_embed_dim,
            'subtoken_embed_dim' : self.args.subtoken_embed_dim,
            'pos_embed_dim' : self.args.pos_embed_dim,
            'rnn_unit_type' : self.args.rnn_unit_type,
            'rnn_bidirection' : self.args.rnn_bidirection,
            'rnn_n_layers' : self.args.rnn_n_layers,
            'rnn_n_units' : self.args.rnn_n_units,
            # 'mlp4pospred_n_layers' : self.args.mlp4pospred_n_layers,
            # 'mlp4pospred_n_units' : self.args.mlp4pospred_n_units,
            'mlp4arcrep_n_layers' : self.args.mlp4arcrep_n_layers,
            'mlp4arcrep_n_units' : self.args.mlp4arcrep_n_units,
            'mlp4labelrep_n_layers' : self.args.mlp4labelrep_n_layers,
            'mlp4labelrep_n_units' : self.args.mlp4labelrep_n_units,
            'mlp4labelpred_n_layers' : self.args.mlp4labelpred_n_layers,
            'mlp4labelpred_n_units' : self.args.mlp4labelpred_n_units,
            'rnn_dropout' : self.args.rnn_dropout,
            'hidden_mlp_dropout' : self.args.hidden_mlp_dropout,
            'pred_layers_dropout' : self.args.pred_layers_dropout,
            'task' : self.args.task,
            'subpos_depth' : self.args.subpos_depth,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
            'token_freq_threshold' : self.args.token_freq_threshold,
            'token_max_vocab_size' : self.args.token_max_vocab_size,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'pretrained_unigram_embed_dim' or
                    key == 'unigram_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'pretrained_unigram_embed_dim' or
                    key == 'pos_embed_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    # key == 'mlp4pospred_n_layers' or
                    # key == 'mlp4pospred_n_units' or
                    key == 'mlp4arcrep_n_layers' or
                    key == 'mlp4arcrep_n_units' or
                    key == 'mlp4labelrep_n_layers' or
                    key == 'mlp4labelrep_n_units' or
                    key == 'mlp4labelpred_n_layers' or
                    key == 'mlp4labelpred_n_units' or
                    key == 'subpos_depth' or
                    key == 'token_freq_threshold' or
                    key == 'token_max_vocab_size'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'hidden_mlp_dropout' or
                      key == 'pred_layers_dropout'
                ):
                    val = float(val)

                elif (key == 'rnn_bidirection' or
                      key == 'lowercase' or
                      key == 'normalize_digits' or
                      key == 'fix_pretrained_embed'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']

        if (self.args.execute_mode != 'interactive' and 
            self.hparams['pretrained_unigram_embed_dim'] > 0 and not self.unigram_embed_model):
            self.log('Error: unigram embedding model is necessary.')
            sys.exit()    

        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_unigram_embed_dim'] != pretrained_unigram_embed_dim:
                self.log(
                    'Error: pretrained_unigram_embed_dim and dimension of loaded embedding model'
                    + 'are conflicted.'.format(
                        hparams['pretrained_unigram_embed_dim'], pretrained_unigram_embed_dim))
                sys.exit()


    def load_data_for_training(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        super().load_data_for_training(refer_tokens=refer_tokens)


    def load_test_data(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        super().load_test_data(refer_tokens=refer_tokens)


    def load_decode_data(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        super().load_decode_data(refer_tokens=refer_tokens)


    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold_head: {} ... {}\n'.format(train.outputs[1][0], train.outputs[1][-1]))
        if self.dic.has_table(constants.ARC_LABEL):
            self.log('# train_gold_label: {} ... {}\n'.format(train.outputs[2][0], train.outputs[2][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        if self.dic.has_table(constants.SUBTOKEN):
            st2i_tmp = list(self.dic.tables[constants.SUBTOKEN].str2id.items())
            self.log('# subtoken2id: {} ... {}\n'.format(st2i_tmp[:10], st2i_tmp[len(st2i_tmp)-10:]))
        if self.dic.has_table(constants.SEG_LABEL):
            id2seg = {v:k for k,v in self.dic.tables[constants.SEG_LABEL].str2id.items()}
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.dic.has_table(constants.POS_LABEL):
            id2pos = {v:k for k,v in self.dic.tables[constants.POS_LABEL].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        if self.dic.has_table(constants.ARC_LABEL):
            id2arc = {v:k for k,v in self.dic.tables[constants.ARC_LABEL].str2id.items()}
            self.log('# arc labels: {}\n'.format(id2arc))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []
        if self.unigram_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_unigram_embed/W')
            self.log('Fix parameters: pretrained_unigram_embed/W')
        if delparams:
            self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_evaluator(self):
        if common.is_typed_parsing_task(self.task) and self.args.ignored_labels:
            tmp = self.args.ignored_labels
            self.args.ignored_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.tables[constants.ARC_LABEL].get_id(label)
                if label_id >= 0:
                    self.args.ignored_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))

        else:
            self.args.ignored_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        ss = ([[xp.asarray(subs, dtype='i') for subs in data.inputs[1][j]] for j in ids] 
              if data.inputs[3] else None)
        ps = ([xp.asarray(data.outputs[0][j], dtype='i') for j in ids] 
              if len(data.outputs) > 0 and data.outputs[0] else None)

        if evaluate:
            ths = [xp.asarray(data.outputs[1][j], dtype='i') for j in ids]
            if common.is_typed_parsing_task(self.task):
                tls = [xp.asarray(data.outputs[2][j], dtype='i') for j in ids]
                return us, ss, ps, ths, tls
            else:
                return us, ss, ps, ths
        else:
            return ws, cs, ps


    def decode(self, rdata, use_pos=False, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        orgseqs = rdata.orgdata[0]
        orgposs = rdata.orgdata[1] if use_pos and len(rdata.orgdata) > 1 else None

        timer = util.Timer()
        timer.start()
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            ws, cs, ps  = self.gen_inputs(rdata, ids, evaluate=False)
            ows = [orgseqs[j] for j in ids]
            ops = [orgposs[j] for j in ids] if orgposs else None
            self.decode_batch(ws, cs, ps, ows, ops, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, ws, cs, ps, orgseqs, orgposs, file=sys.stdout):
        use_pos = ps is not None
        label_prediction = common.is_typed_parsing_task(self.task)
        
        n_ins = len(ws)
        ret = self.classifier.decode(ws, cs, ps, label_prediction)

        hs = ret[0]
        if label_prediction:
            ls = ret[1]
            id2label = self.dic.tables[constants.ARC_LABEL].id2str
        else:
            ls = [None] * len(ws)
        if not use_pos:
            ps = [None] * len(ws)
            orgposs = [None] * len(ws)

        for x_str, p_str, h, l in zip(orgseqs, orgposs, hs, ls):
            x_str = x_str[1:]
            p_str = p_str[1:] if p_str else None
            h = h[1:]
            if label_prediction:
                l = l[1:]
                l_str = [id2label[int(li)] for li in l] 

            if label_prediction:
                if use_pos:
                    res = ['{}\t{}\t{}\t{}'.format(
                        xi_str, pi_str, hi, li_str) for xi_str, pi_str, hi, li_str in zip(
                            x_str, p_str, h, l_str)]
                        
                else:
                    res = ['{}\t{}\t{}'.format(
                        xi_str, hi, li_str) for xi_str, hi, li_str in zip(x_str, h, l_str)]
            else:
                if use_pos:
                    res = ['{}\t{}\t{}'.format(
                        xi_str, pi_str, hi) for xi_str, pi_str, hi in zip(x_str, p_str, h)]
                else:
                    res = ['{}\t{}'.format(
                        xi_str, hi) for xi_str, hi in zip(x_str, h)]
            res = '\n'.join(res).rstrip()

            print(res+'\n', file=file)


    def run_interactive_mode(self):
        use_pos = constants.POS_LABEL in self.dic.tables
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False
        use_subtoken = ('subtoken_embed_dim' in self.hparams and self.hparams['subtoken_embed_dim'] > 0)
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = data_io.parse_commandline_input(
                line, self.dic, self.task, use_pos=use_pos and '/' in line,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=use_subtoken, subpos_depth=subpos_depth)

            ws, cs, ps = self.gen_inputs(rdata, [0], evaluate=False)
            orgseqs = rdata.orgdata[0]
            orgposs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(ws, cs, ps, orgseqs, orgposs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)
