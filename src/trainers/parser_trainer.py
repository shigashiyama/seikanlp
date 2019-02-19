import sys

from chainer import cuda

import numpy as np

import classifiers.dependency_parser
import common
import constants
from data_loaders import data_loader, parsing_data_loader
from evaluators.parser_evaluator import ParserEvaluator, TypedParserEvaluator
import models.parser
import models.util
from trainers import trainer
from trainers.trainer import Trainer
import util


class ParserTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.label_begin_index = 2


    def show_data_info(self, data_type):
        dic = self.dic_dev if data_type == 'devel' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Num of tokens: {}'.format(len(dic.tables[constants.UNIGRAM])))
        

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

        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        for i in range(len(attr_indexes)):
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                id2attr = {v:k for k,v in self.dic.tables[constants.ATTR_LABEL(i)].str2id.items()}
                self.log('# {}-th attribute labels: {}\n'.format(i, id2attr))
        if self.dic.has_table(constants.ARC_LABEL):
            id2arc = {v:k for k,v in self.dic.tables[constants.ARC_LABEL].str2id.items()}
            self.log('# arc labels: {}\n'.format(id2arc))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(self.args.unigram_embed_model_path)


    def init_model(self):
        super().init_model()
        if self.unigram_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, finetuning=True)


    def load_model(self):
        super().load_model()
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
        if 'pred_layers_dropout' in self.hparams:
            self.classifier.change_pred_layers_dropout_ratio(self.hparams['pred_layers_dropout'])
            

    def update_model(self, classifier=None, dic=None, train=False):
        if not classifier:
            classifier = self.classifier
        if not dic:
            dic = self.dic

        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            classifier.grow_embedding_layers(dic, self.unigram_embed_model, train=train)
            classifier.grow_inference_layers(dic)


    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'unigram_embed_dim' : self.args.unigram_embed_dim,
            'attr1_embed_dim' : self.args.attr1_embed_dim,
            'rnn_unit_type' : self.args.rnn_unit_type,
            'rnn_bidirection' : self.args.rnn_bidirection,
            'rnn_n_layers' : self.args.rnn_n_layers,
            'rnn_n_units' : self.args.rnn_n_units,
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
            'lowercasing' : self.args.lowercasing,
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
                    key == 'pretrained_unigram_embed_dim' or
                    key == 'attr1_embed_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp4arcrep_n_layers' or
                    key == 'mlp4arcrep_n_units' or
                    key == 'mlp4labelrep_n_layers' or
                    key == 'mlp4labelrep_n_units' or
                    key == 'mlp4labelpred_n_layers' or
                    key == 'mlp4labelpred_n_units' or
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
                      key == 'lowercasing' or
                      key == 'normalize_digits'
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
                    + ' are conflicted.'.format(
                        hparams['pretrained_unigram_embed_dim'], pretrained_unigram_embed_dim))
                sys.exit()


    def setup_data_loader(self):
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        self.data_loader = parsing_data_loader.ParsingDataLoader(
            token_index=self.args.token_index,
            head_index=self.args.head_index,
            arc_index=self.args.arc_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
            attr_chunking_flags=common.get_attribute_boolvalues(
                self.args.attr_chunking_flags, len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            attr_delim=self.args.attr_delim,
            use_arc_label=(self.task == constants.TASK_TDEP),
            lowercasing=self.hparams['lowercasing'],
            normalize_digits=self.hparams['normalize_digits'],
            token_freq_threshold=self.hparams['token_freq_threshold'],
            token_max_vocab_size=self.hparams['token_max_vocab_size'],
            unigram_vocab=(self.unigram_embed_model.wv if self.unigram_embed_model else set()),
        )


    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_dim = hparams['unigram_embed_dim']

        if 'pretrained_unigram_embed_dim' in hparams and hparams['pretrained_unigram_embed_dim'] > 0:
            pretrained_unigram_embed_dim = hparams['pretrained_unigram_embed_dim']
        else:
            pretrained_unigram_embed_dim = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        n_attr1 = len(dic.tables[constants.ATTR_LABEL(0)]) if (
            hparams['attr1_embed_dim'] > 0 and constants.ATTR_LABEL(0) in dic.tables) else 0
        n_labels = len(dic.tables[constants.ARC_LABEL]) if common.is_typed_parsing_task(self.task) else 0
        attr1_embed_dim = hparams['attr1_embed_dim'] if n_attr1 > 0 else 0

        if (pretrained_embed_usage == models.util.ModelUsage.ADD or
            pretrained_embed_usage == models.util.ModelUsage.INIT):
            if pretrained_unigram_embed_dim > 0 and pretrained_unigram_embed_dim != unigram_embed_dim:
                print('Error: pre-trained and random initialized unigram embedding vectors '
                      + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                      + ': d1={}, d2={}'.format(pretrained_unigram_embed_dim, unigram_embed_dim),
                      file=sys.stderr)
                sys.exit()

        predictor = models.parser.RNNBiaffineParser(
            n_vocab, unigram_embed_dim, n_attr1, attr1_embed_dim,
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], hparams['rnn_n_layers'], 
            hparams['rnn_n_units'], 
            hparams['mlp4arcrep_n_layers'], hparams['mlp4arcrep_n_units'],
            hparams['mlp4labelrep_n_layers'], hparams['mlp4labelrep_n_units'],
            mlp4labelpred_n_layers=hparams['mlp4labelpred_n_layers'], 
            mlp4labelpred_n_units=hparams['mlp4labelpred_n_units'],
            n_labels=n_labels, rnn_dropout=hparams['rnn_dropout'], 
            hidden_mlp_dropout=hparams['hidden_mlp_dropout'], 
            pred_layers_dropout=hparams['pred_layers_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage)

        self.classifier = classifiers.dependency_parser.DependencyParser(predictor)



    def setup_evaluator(self, evaluator=None):
        if common.is_typed_parsing_task(self.task) and self.args.ignored_labels:
            ignored_labels = set()
            for label in self.args.ignored_labels.split(','):
                label_id = self.dic.tables[constants.ARC_LABEL].get_id(label)
                if label_id >= 0:
                    ignored_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(ignored_labels))

        else:
            ignored_labels = set()

        evaluator = None
        if self.task == constants.TASK_DEP:
            evaluator = ParserEvaluator(ignore_head=True)
        
        elif self.task == constants.TASK_TDEP:
            evaluator = TypedParserEvaluator(
                ignore_head=True, ignored_labels=ignored_labels)

        self.evaluator = evaluator


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np
        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        ps = ([xp.asarray(data.outputs[0][j], dtype='i') for j in ids] 
              if len(data.outputs) > 0 and data.outputs[0] else None)

        if evaluate:
            ths = [xp.asarray(data.outputs[1][j], dtype='i') for j in ids]
            if common.is_typed_parsing_task(self.task):
                tls = [xp.asarray(data.outputs[2][j], dtype='i') for j in ids]
                return us, ps, ths, tls
            else:
                return us, ps, ths
        else:
            return us, ps


    def decode(self, rdata, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        org_tokens = rdata.orgdata[0]
        org_attrs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

        timer = util.Timer()
        timer.start()
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            inputs = self.gen_inputs(rdata, ids, evaluate=False)
            ot = [org_tokens[j] for j in ids]
            oa = [org_attrs[j] for j in ids] if org_attrs else None
            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, *inputs, org_tokens=None, org_attrs=None, file=sys.stdout):
        label_prediction = self.task == constants.TASK_TDEP
        ret = self.classifier.decode(*inputs, label_prediction=label_prediction)
        hs = ret[0]
        if label_prediction:
            ls = ret[1]
            id2label = self.dic.tables[constants.ARC_LABEL].id2str
        else:
            ls = [None] * len(org_tokens)
        if not org_attrs:
            org_attrs = [None] * len(org_tokens)

        for x_str, p_str, h, l in zip(org_tokens, org_attrs, hs, ls):
            x_str = x_str[1:]
            p_str = p_str[1:] if p_str else None
            h = h[1:]

            if label_prediction:
                l = l[1:]
                l_str = [id2label[int(li)] for li in l] 

            if label_prediction:
                if p_str:
                    res = ['{}\t{}\t{}\t{}\t{}'.format(
                        i+1, xi_str, pi_str, hi, li_str) for i, (xi_str, pi_str, hi, li_str) in enumerate(zip(
                            x_str, p_str, h, l_str))]
                        
                else:
                    res = ['{}\t{}\t{}\t{}'.format(
                        i+1, xi_str, hi, li_str) for i, (xi_str, hi, li_str) in enumerate(zip(x_str, h, l_str))]
            else:
                if p_str:
                    res = ['{}\t{}\t{}\t{}'.format(
                        i+1, xi_str, pi_str, hi) for i, (xi_str, pi_str, hi) in enumerate(zip(x_str, p_str, h))]
                else:
                    res = ['{}\t{}\t{}'.format(
                        i+1, xi_str, hi) for i, (xi_str, hi) in enumerate(zip(x_str, h))]

            res = '\n'.join(res).rstrip()
            print(res+'\n', file=file)


    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)
