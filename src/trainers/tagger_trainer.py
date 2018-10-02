import sys
import pickle
from collections import Counter # tmp

import numpy as np

import chainer
from chainer import cuda

import classifiers.sequence_tagger
import common
import constants
from data import data_loader, segmentation_data_loader, tagging_data_loader
import evaluators
import models.tagger
import models.util
import optimizers
from trainers import trainer
from trainers.trainer import Trainer
import util

class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def show_data_info(self, data_type):
        dic = self.dic_dev if data_type == 'devel' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Num of tokens: {}'.format(len(dic.tables[constants.UNIGRAM])))
        if dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(len(dic.tables[constants.BIGRAM])))
        if dic.has_table(constants.TOKEN_TYPE):
            self.log('Num of tokentypes: {}'.format(len(dic.tables[constants.TOKEN_TYPE])))
        if dic.has_table(constants.SUBTOKEN):
            self.log('Num of subtokens: {}'.format(len(dic.tables[constants.SUBTOKEN])))
        self.log()
        

    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold: {} ... {}\n'.format(train.outputs[0][0], train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))
        if self.dic.has_table(constants.BIGRAM):
            b2i_tmp = list(self.dic.tables[constants.BIGRAM].str2id.items())
            self.log('# bigram2id: {} ... {}\n'.format(b2i_tmp[:10], b2i_tmp[len(b2i_tmp)-10:]))
        if self.dic.has_table(constants.TOKEN_TYPE):
            tt2i = list(self.dic.tables[constants.TOKEN_TYPE].str2id.items())
            self.log('# tokentype2id: {}\n'.format(tt2i))
        if self.dic.has_table(constants.SUBTOKEN):
            st2i_tmp = list(self.dic.tables[constants.SUBTOKEN].str2id.items())
            self.log('# subtoken2id: {} ... {}\n'.format(st2i_tmp[:10], st2i_tmp[len(st2i_tmp)-10:]))
        if self.dic.has_trie(constants.CHUNK):
            id2chunk = self.dic.tries[constants.CHUNK].id2chunk
            n_chunks = len(self.dic.tries[constants.CHUNK])
            c2i_head = [(id2chunk[i], i) for i in range(0, min(10, n_chunks))]
            c2i_tail = [(id2chunk[i], i) for i in range(max(0, n_chunks-10), n_chunks)]
            self.log('# chunk2id: {} ... {}\n'.format(c2i_head, c2i_tail))
        if self.dic.has_table(constants.SEG_LABEL):
            id2seg = {v:k for k,v in self.dic.tables[constants.SEG_LABEL].str2id.items()}
            self.log('# label_set: {}\n'.format(id2seg))

        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        for i in range(len(attr_indexes)):
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                id2attr = {v:k for k,v in self.dic.tables[constants.ATTR_LABEL(i)].str2id.items()}
                self.log('# {}-th attribute labels: {}\n'.format(i, id2attr))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        ts = [xp.asarray(data.inputs[2][j], dtype='i') for j in ids] if data.inputs[2] else None
        es = [xp.asarray(data.inputs[3][j], dtype='i') for j in ids] if data.inputs[3] else None
        fs = [data.featvecs[j] for j in ids] if self.hparams['feature_template'] else None
        ls = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None

        if evaluate:
            return us, bs, ts, es, fs, ls
        else:
            return us, bs, ts, es, fs
        

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


    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            # if self.hparams['feature_template']:
            #     rdata.featvecs = self.feat_extractor.extract_features(ws[0], self.dic.tries[constants.CHUNK])
            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


    def run_eval_mode2(self):
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)

        id2token = self.dic.tables[constants.UNIGRAM].id2str
        id2label = self.dic.tables[constants.SEG_LABEL].id2str

        data = self.test
        n_ins = len(data.inputs[0])
        n_sen = 0
        total_counts = None
                
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            n_sen += len(xs)
            golds = inputs[self.label_begin_index:]

            with chainer.no_backprop_mode():
                gls, pls = classifier.predictor.do_analysis(*inputs)

            for gl, pl in zip(gls, pls):
                for gli, pli in zip(gl, pl):
                    print('{}'.format(int(gli) == int(pli)))
                print()

        print('Finished', n_sen)


    def decode_batch(self, *inputs, org_tokens, org_attrs, file=sys.stdout):
        ys = self.classifier.decode(*inputs)
        id2label = (self.dic.tables[constants.SEG_LABEL if common.is_segmentation_task(self.task) 
                                    else constants.ATTR_LABEL(0)].id2str)
        if not org_attrs:
            org_attrs = [None] * len(org_tokens)

        for x_str, a_str, y in zip(org_tokens, org_attrs, ys):
            y_str = [id2label[int(yi)] for yi in y]

            if self.task == constants.TASK_TAG:
                if a_str:
                    res = ['{}{}{}{}{}'.format(xi_str, self.args.output_attr_delim,
                                               ai_str, self.args.output_attr_delim,
                                               yi_str) 
                           for xi_str, ai_str, yi_str in zip(x_str, a_str, y_str)]
                else:
                    res = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                           for xi_str, yi_str in zip(x_str, y_str)]
                
                res = self.args.output_token_delim.join(res)

            elif self.task == constants.TASK_SEG:
                res = ['{}{}'.format(xi_str, self.args.output_token_delim 
                                     if (yi_str.startswith('E') or yi_str.startswith('S')) 
                                     else '') for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip(' ')

            elif self.task == constants.TASK_SEGTAG:
                res = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip(' ')

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            print(res, file=file)
            if self.args.output_empty_line:
                print(file=file)


    def load_external_dictionary(self):
        if not self.dic and self.args.external_dic_path:
            use_pos = common.is_tagging_task(self.task)
            edic_path = self.args.external_dic_path
            self.dic = data_loader.load_external_dictionary(edic_path, use_pos=use_pos)
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
            
            if not edic_path.endswith('pickle'):
                base = edic_path.split('.')[0]
                edic_pic_path = base + '.pickle'
                with open(edic_pic_path, 'wb') as f:
                    pickle.dump(self.dic, f)
                self.log('Dumpped dictionary: {}'.format(edic_pic_path))
            self.log('')


class TaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.label_begin_index = 5


    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(self.args.unigram_embed_model_path)
        if self.args.bigram_embed_model_path:
            self.bigram_embed_model = trainer.load_embedding_model(self.args.bigram_embed_model_path)


    def init_model(self):
        super().init_model()
        if self.unigram_embed_model or self.bigram_embed_model:
            finetuning = not self.hparams['fix_pretrained_embed']
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, self.bigram_embed_model, finetuning=finetuning)


    def load_model(self):
        super().load_model()
        if self.args.execute_mode == 'train':
            if 'embed_dropout' in self.hparams:
                self.classifier.change_embed_dropout_ratio(self.hparams['embed_dropout'])
            if 'rnn_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
            if 'hidden_mlp_dropout' in self.hparams:
                self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
        else:
            self.classifier.change_dropout_ratio(0)


    def update_model(self, classifier=None, dic=None):
        if not classifier:
            classifier = self.classifier
        if not dic:
            dic = self.dic

        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            classifier.grow_embedding_layers(
                dic, self.unigram_embed_model, self.bigram_embed_model,
                train=(self.args.execute_mode=='train'))
            classifier.grow_inference_layers(dic)


    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        if self.bigram_embed_model:
            pretrained_bigram_embed_dim = self.bigram_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_bigram_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_bigram_embed_dim' : pretrained_bigram_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'fix_pretrained_embed' : self.args.fix_pretrained_embed,
            'unigram_embed_dim' : self.args.unigram_embed_dim,
            'bigram_embed_dim' : self.args.bigram_embed_dim,
            'tokentype_embed_dim' : self.args.tokentype_embed_dim,
            'subtoken_embed_dim' : self.args.subtoken_embed_dim,
            'attr1_embed_dim' : self.args.attr1_embed_dim,
            'rnn_unit_type' : self.args.rnn_unit_type,
            'rnn_bidirection' : self.args.rnn_bidirection,
            'rnn_n_layers' : self.args.rnn_n_layers,
            'rnn_n_units' : self.args.rnn_n_units,
            'mlp_n_layers' : self.args.mlp_n_layers,
            'mlp_n_units' : self.args.mlp_n_units,
            'inference_layer' : self.args.inference_layer,
            'embed_dropout' : self.args.embed_dropout,
            'rnn_dropout' : self.args.rnn_dropout,
            'mlp_dropout' : self.args.mlp_dropout,
            'feature_template' : self.args.feature_template,
            'task' : self.args.task,
            'tagging_scheme': self.args.tagging_scheme,
            'lowercasing' : self.args.lowercasing,
            'normalize_digits' : self.args.normalize_digits,
            'token_freq_threshold' : self.args.token_freq_threshold,
            'token_max_vocab_size' : self.args.token_max_vocab_size,
            'bigram_freq_threshold' : self.args.bigram_freq_threshold,
            'bigram_max_vocab_size' : self.args.bigram_max_vocab_size,
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
                    key == 'pretrained_bigram_embed_dim' or
                    key == 'unigram_embed_dim' or
                    key == 'bigram_embed_dim' or
                    key == 'tokentype_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'attr1_embed_dim' or
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp_n_layers'or
                    key == 'mlp_n_units' or
                    key == 'token_freq_threshold' or
                    key == 'token_max_vocab_size' or
                    key == 'bigram_freq_threshold' or
                    key == 'bigram_max_vocab_size' or
                    key == 'chunk_freq_threshold' or
                    key == 'chunk_max_vocab_size'
                ):
                    val = int(val)

                elif (key == 'embed_dropout' or
                      key == 'rnn_dropout' or
                      key == 'mlp_dropout'
                ):
                    val = float(val)

                elif (key == 'rnn_bidirection' or
                      key == 'lowercasing' or
                      key == 'normalize_digits' or
                      key == 'fix_pretrained_embed'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        # tmp
        # if not 'attr1_embed_dim' in hparams:
        #     hparams['attr1_embed_dim'] = 0
        # if not 'tagging_scheme'in hparams:
        #     hparams['tagging_scheme'] = 'BIOES'

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

        if self.bigram_embed_model:
            pretrained_bigram_embed_dim = self.bigram_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_bigram_embed_dim'] != pretrained_bigram_embed_dim:
                self.log(
                    'Error: pretrained_bigram_embed_dim and dimension of loaded embedding model'
                    + 'are conflicted.'.format(
                        hparams['pretrained_bigram_embed_dim'], pretrained_bigram_embed_dim))
                sys.exit()


    def setup_data_loader(self):
        # scheme = hparams['tagging_scheme']
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)

        if self.task == constants.TASK_SEG:
            self.data_loader = segmentation_data_loader.SegmentationDataLoader(
                token_index=self.args.token_index,
                attr_indexes=attr_indexes,
                attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
                attr_target_labelsets=common.get_attribute_labelsets(
                    self.args.attr_target_labelsets, len(attr_indexes)),
                attr_delim=self.args.attr_delim,
                use_bigram=(self.hparams['bigram_embed_dim'] > 0),
                use_tokentype=(self.hparams['tokentype_embed_dim'] > 0),
                use_chunk_trie=(True if self.hparams['feature_template'] else False),
                bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
                bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
                unigram_vocab=(self.unigram_embed_model.wv if self.unigram_embed_model else set()),
                bigram_vocab=(self.bigram_embed_model.wv if self.bigram_embed_model else set()),
            )
        else:
            self.data_loader = tagging_data_loader.TaggingDataLoader(
                token_index=self.args.token_index,
                attr_indexes=attr_indexes,
                attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
                attr_chunking_flags=common.get_attribute_boolvalues(
                    self.args.attr_chunking_flags, len(attr_indexes)),
                attr_target_labelsets=common.get_attribute_labelsets(
                    self.args.attr_target_labelsets, len(attr_indexes)),
                attr_delim=self.args.attr_delim,
                use_subtoken=(self.hparams['subtoken_embed_dim'] > 0),
                lowercasing=self.hparams['lowercasing'],
                normalize_digits=self.hparams['normalize_digits'],
                token_freq_threshold=self.hparams['token_freq_threshold'],
                token_max_vocab_size=self.hparams['token_max_vocab_size'],
                unigram_vocab=(self.unigram_embed_model.wv if self.unigram_embed_model else set()),
            )


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []

        if self.unigram_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_unigram_embed/W')
        if self.bigram_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_bigram_embed/W')

        if delparams:
            for params in delparams:
                self.log('Fix parameters: {}'.foramt(param))                
            self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_dim = hparams['unigram_embed_dim']
        
        if 'bigram_embed_dim' in hparams and hparams['bigram_embed_dim'] > 0:
            bigram_embed_dim = hparams['bigram_embed_dim']
            n_bigrams = len(dic.tables[constants.BIGRAM])
        else:
            bigram_embed_dim = n_bigrams = 0

        if 'tokentype_embed_dim' in hparams and hparams['tokentype_embed_dim'] > 0:
            tokentype_embed_dim = hparams['tokentype_embed_dim']
            n_tokentypes = len(dic.tables[constants.TOKEN_TYPE])
        else:
            tokentype_embed_dim = n_tokentypes = 0

        subtoken_embed_dim = n_subtokens = 0

        if 'pretrained_unigram_embed_dim' in hparams and hparams['pretrained_unigram_embed_dim'] > 0:
            pretrained_unigram_embed_dim = hparams['pretrained_unigram_embed_dim']
        else:
            pretrained_unigram_embed_dim = 0

        if 'pretrained_bigram_embed_dim' in hparams and hparams['pretrained_bigram_embed_dim'] > 0:
            pretrained_bigram_embed_dim = hparams['pretrained_bigram_embed_dim']
        else:
            pretrained_bigram_embed_dim = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        if common.is_segmentation_task(self.task):
            n_label = len(dic.tables[constants.SEG_LABEL])
            n_labels = [n_label]
            attr1_embed_dim = n_attr1 = 0

        else:
            n_labels = []
            for i in range(3): # tmp
                if constants.ATTR_LABEL(i) in dic.tables:
                    n_label = len(dic.tables[constants.ATTR_LABEL(i)])
                    n_labels.append(n_label)
                
            if 'attr1_embed_dim' in hparams and hparams['attr1_embed_dim'] > 0:
                attr1_embed_dim = hparams['attr1_embed_dim']
                n_attr1 = n_labels[1] if len(n_labels) > 1 else 0
            else:
                attr1_embed_dim = n_attr1 = 0

        if (pretrained_embed_usage == models.util.ModelUsage.ADD or
            pretrained_embed_usage == models.util.ModelUsage.INIT):
            if pretrained_unigram_embed_dim > 0 and pretrained_unigram_embed_dim != unigram_embed_dim:
                print('Error: pre-trained and random initialized unigram embedding vectors '
                      + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                      + ': d1={}, d2={}'.format(pretrained_unigram_embed_dim, unigram_embed_dim),
                      file=sys.stderr)
                sys.exit()

            if pretrained_bigram_embed_dim > 0 and pretrained_bigram_embed_dim != bigram_embed_dim:
                print('Error: pre-trained and random initialized bigram embedding vectors '
                      + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                      + ': d1={}, d2={}'.format(pretrained_bigram_embed_dim, bigram_embed_dim),
                      file=sys.stderr)
                sys.exit()

        predictor = models.tagger.construct_RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_attr1, attr1_embed_dim, 0, 0,
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], 
            hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['rnn_n_layers2'] if 'rnn_n_layers2' in hparams else 0,
            hparams['rnn_n_units2'] if 'rnn_n_units2' in hparams else 0,
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels[0], 
            use_crf=hparams['inference_layer'] == 'crf',
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=0,
            rnn_dropout=hparams['rnn_dropout'],
            embed_dropout=hparams['embed_dropout'] if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage)

        self.classifier = classifiers.sequence_tagger.SequenceTagger(predictor, task=self.task)


    def setup_evaluator(self):
        if self.task == constants.TASK_TAG and self.args.ignored_labels: # TODO fix
            tmp = self.args.ignored_labels
            self.args.ignored_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.tables[constants.ATTR_LABEL(0)].get_id(label)
                if label_id >= 0:
                    self.args.ignored_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))

        else:
            self.args.ignored_labels = set()

        # TODO reflect ignored_labels
        evaluator = None
        if self.task == constants.TASK_SEG:
            evaluator = evaluators.FMeasureEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)

        elif self.task == constants.TASK_SEGTAG:
            evaluator = evaluators.DoubleFMeasureEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)

        elif self.task == constants.TASK_TAG:
            if common.use_fmeasure(self.dic.tables[constants.ATTR_LABEL(0)].str2id):
                evaluator = evaluators.FMeasureEvaluator(self.dic.tables[constants.ATTR_LABEL(0)].id2str)
            else:
                evaluator = evaluators.AccuracyEvaluator(self.dic.tables[constants.ATTR_LABEL(0)].id2str)
        self.evaluator = evaluator
        self.evaluator.calculator.id2token = self.dic.tables[constants.UNIGRAM].id2str # tmp


    # tmp
    def run_train_mode2(self):
        id2label = self.dic.tables[constants.SEG_LABEL].id2str
        uni2label_tr = None
        bi2label_tr = None

        for i in range(2):
            print('<train>' if i == 0 else '<devel>')
            data = self.train if i == 0 else self.dev
            labels = Counter()
            uni2label = models.attribute_annotator.Key2Counter()
            bi2label = models.attribute_annotator.Key2Counter()
            n_tokens = 0

            n_ins = len(data.inputs[0])
            ids = [i for i in range(n_ins)]
            inputs = self.gen_inputs(data, ids)
            us = inputs[0]
            bs = inputs[1]
            ls = inputs[self.label_begin_index]

            # count
            for u, b, l in zip(us, bs, ls):
                n_tokens += len(u)
                for ui, bi, li in zip(u, b, l):
                    labels[li] += 1
                    uni2label.add(ui, li)
                    bi2label.add(bi, li)

            # calc label distribution
            for lab, cnt in labels.items():
                rate = cnt / n_tokens
                print(id2label[lab], rate*100)

            # calc entropy for unigram
            u_unk = 0
            u_ent = 0
            for u, counter in uni2label.key2counter.items():
                total = sum([v for v in counter.values()])
                ent = 0
                for key, count in counter.items():
                    p = count / total
                    ent += - p * np.log2(p)
                u_ent += ent
                if u == 0:
                    u_unk = total
                    print('unigram entropy (unk)', ent)

            u_ent /= len(uni2label)
            print('unigram entropy', u_ent)
            print('unigram unk ratio', u_unk / n_tokens)

            # calc entropy for bigram
            b_unk = 0
            b_ent = 0
            for b, counter in bi2label.key2counter.items():
                total = sum([v for v in counter.values()])
                ent = 0
                for key, count in counter.items():
                    p = count / total
                    ent += - p * np.log2(p)
                b_ent += ent
                if b == 0:
                    b_unk = total
                    print('bigram entropy (unk)', ent)


            b_ent /= len(bi2label)
            print('bigram entropy', b_ent)
            print('bigram unk ratio', b_unk / n_tokens)

            # evaluate using unigram
            if i == 0:
                uni2label_tr = uni2label
                bi2label_tr = bi2label
                continue

            os = []
            for u in us:
                o = []
                for ui in u:
                    counter = uni2label_tr.get(ui)
                    if counter:
                        pred = counter.most_common(1)[0][0]
                    else:
                        pred = -1
                    o.append(pred)
                os.append(o)
                
            counts = self.evaluator.calculate(*[us], *[ls], *[os])
            res = self.evaluator.report_results(n_ins, counts, 0.0)

            # evaluate using bigram
            os = []
            for b in bs:
                o = []
                for bi in b:
                    counter = bi2label_tr.get(bi)
                    if counter:
                        pred = counter.most_common(1)[0][0]
                    else:
                        pred = -1
                    o.append(pred)
                os.append(o)
                
            counts = self.evaluator.calculate(*[bs], *[ls], *[os])
            res = self.evaluator.report_results(n_ins, counts, 0.0)

            print()
