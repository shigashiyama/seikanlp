import sys
from collections import Counter # tmp

import numpy as np

import chainer
from chainer import cuda

import classifiers.hybrid_sequence_tagger
import common
import constants
from data_loaders import segmentation_data_loader, tagging_data_loader
import dictionary
from evaluators.hybrid_unit_segmenter_evaluator import HybridSegmenterEvaluator, HybridTaggerEvaluator, AccuracyEvaluatorForAttention
from evaluators.tagger_evaluator import FMeasureEvaluatorForEachVocab
import models.tagger
import models.util
from trainers import trainer
from trainers.tagger_trainer import TaggerTrainerBase


def get_num_candidates(mask_pairs, n_tokens):
    ncand = [0] * n_tokens
    for i, j in mask_pairs:
        ncand[i] += 1
    return ncand


# character word hybrid segmenter/tagger
class HybridUnitSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.chunk_embed_model = None
        self.label_begin_index = 6


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
            segmentation_data_loader.add_chunk_sequences(
                rdata, self.dic,
                max_len=self.hparams['max_chunk_len'], evaluate=False,
                model_type=self.hparams['chunk_pooling_type'])

            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)


    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(self.args.unigram_embed_model_path)
        if self.args.bigram_embed_model_path:
            self.bigram_embed_model = trainer.load_embedding_model(self.args.bigram_embed_model_path)
        if self.args.chunk_embed_model_path:
            self.chunk_embed_model = trainer.load_embedding_model(self.args.chunk_embed_model_path)


    def init_model(self):
        super().init_model()
        if self.unigram_embed_model or self.bigram_embed_model or self.chunk_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, self.bigram_embed_model, self.chunk_embed_model, True)


    def load_model(self):
        super().load_model()
        if self.args.execute_mode == 'train':
            if 'embed_dropout' in self.hparams:
                self.classifier.change_embed_dropout_ratio(self.hparams['embed_dropout'])
            if 'rnn_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
            if 'hidden_mlp_dropout' in self.hparams:
                self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])
            if 'biaffine_dropout' in self.hparams:
                self.classifier.change_biaffine_dropout_ratio(self.hparams['biaffine_dropout'])
            if 'chunk_vector_dropout' in self.hparams:
                self.classifier.change_chunk_vector_dropout_ratio(self.hparams['chunk_vector_dropout'])
        else:
            self.classifier.change_dropout_ratio(0)


    def update_model(self, classifier=None, dic=None, train=False):
        if not classifier:
            classifier = self.classifier
        if not dic:
            dic = self.dic

        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            classifier.grow_embedding_layers(
                dic, self.unigram_embed_model, self.bigram_embed_model, self.chunk_embed_model,
                train=(self.args.execute_mode=='train'), fasttext=self.args.gen_oov_chunk_for_test)
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

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_chunk_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_bigram_embed_dim' : pretrained_bigram_embed_dim,
            'pretrained_chunk_embed_dim' : pretrained_chunk_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'chunk_loss_ratio' : self.args.chunk_loss_ratio,
            'chunk_pooling_type' : self.args.chunk_pooling_type.upper(),
            'biaffine_type' : self.args.biaffine_type,
            'use_gold_chunk' : self.args.use_gold_chunk,
            'unuse_nongold_chunk' : self.args.unuse_nongold_chunk,
            'use_unknown_pretrained_chunk' : not self.args.ignore_unknown_pretrained_chunk,
            'min_chunk_len' : self.args.min_chunk_len,
            'max_chunk_len' : self.args.max_chunk_len,
            'unigram_embed_dim' : self.args.unigram_embed_dim,
            'bigram_embed_dim' : self.args.bigram_embed_dim,
            'chunk_embed_dim' : self.args.chunk_embed_dim,
            'rnn_unit_type' : self.args.rnn_unit_type,
            'rnn_bidirection' : self.args.rnn_bidirection,
            'rnn_n_layers' : self.args.rnn_n_layers,
            'rnn_n_units' : self.args.rnn_n_units,
            'rnn_n_layers2' : self.args.rnn_n_layers2,
            'rnn_n_units2' : self.args.rnn_n_units2,
            'mlp_n_layers' : self.args.mlp_n_layers,
            'mlp_n_units' : self.args.mlp_n_units,
            'inference_layer' : self.args.inference_layer,
            'embed_dropout' : self.args.embed_dropout,            
            'rnn_dropout' : self.args.rnn_dropout,
            'biaffine_dropout' : self.args.biaffine_dropout,
            'chunk_vector_dropout' : self.args.chunk_vector_dropout,
            'chunk_vector_element_dropout' : self.args.chunk_vector_element_dropout,
            'mlp_dropout' : self.args.mlp_dropout,
            'feature_template' : self.args.feature_template,
            'task' : self.args.task,
            'tagging_unit': self.args.tagging_unit,
            'token_freq_threshold' : self.args.token_freq_threshold,
            'token_max_vocab_size' : self.args.token_max_vocab_size,
            'bigram_freq_threshold' : self.args.bigram_freq_threshold,
            'bigram_max_vocab_size' : self.args.bigram_max_vocab_size,
            'chunk_freq_threshold' : self.args.chunk_freq_threshold,
            'chunk_max_vocab_size' : self.args.chunk_max_vocab_size,
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
                    key == 'pretrained_chunk_embed_dim' or
                    key == 'min_chunk_len' or
                    key == 'max_chunk_len' or
                    key == 'unigram_embed_dim' or
                    key == 'bigram_embed_dim' or
                    key == 'chunk_embed_dim' or
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'rnn_n_layers2' or
                    key == 'rnn_n_units2' or
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

                elif (key == 'chunk_loss_ratio' or
                      key == 'embed_dropout' or
                      key == 'rnn_dropout' or
                      key == 'biaffine_dropout' or
                      key == 'chunk_vector_dropout' or
                      key == 'chunk_vector_element_dropout' or
                      key == 'mlp_dropout'
                ):
                    val = float(val)
                    
                elif (key == 'rnn_bidirection' or
                      key == 'lowercase' or
                      key == 'normalize_digits' or
                      key == 'use_gold_chunk' or
                      key == 'unuse_nongold_chunk' or
                      key == 'ignore_unknown_pretrained_chunk'

                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        if not 'unuse_nongold_chunk'in hparams:
            hparams['unuse_nongold_chunk'] = False

        self.hparams = hparams
        self.task = hparams['task']

        if self.args.execute_mode != 'interactive':
            if self.hparams['pretrained_unigram_embed_dim'] > 0 and not self.unigram_embed_model:
                self.log('Error: unigram embedding model is necessary.')
                sys.exit()

            if self.hparams['pretrained_chunk_embed_dim'] > 0 and not self.chunk_embed_model:
                self.log('Error: chunk embedding model is necessary.')
                sys.exit()

        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_unigram_embed_dim'] != pretrained_unigram_embed_dim:
                self.log(
                    'Error: pretrained_unigram_embed_dim and dimension of loaded embedding model'
                    + ' are conflicted.'.format(
                        hparams['pretrained_unigram_embed_dim'], pretrained_unigram_embed_dim))
                sys.exit()

        if self.bigram_embed_model:
            pretrained_bigram_embed_dim = self.bigram_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_bigram_embed_dim'] != pretrained_bigram_embed_dim:
                self.log(
                    'Error: pretrained_bigram_embed_dim and dimension of loaded embedding model'
                    + ' are conflicted.'.format(
                        hparams['pretrained_bigram_embed_dim'], pretrained_bigram_embed_dim))
                sys.exit()

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_chunk_embed_dim'] != pretrained_chunk_embed_dim:
                self.log(
                    'Error: pretrained_chunk_embed_dim and dimension of loaded embedding model'
                    + ' are conflicted.'.format(
                        hparams['pretrained_chunk_embed_dim'], pretrained_chunk_embed_dim))
                sys.exit()


    def setup_data_loader(self):
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)

        self.data_loader = segmentation_data_loader.SegmentationDataLoader(
            token_index=self.args.token_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            use_bigram=(self.hparams['bigram_embed_dim'] > 0),
            use_chunk_trie=True,
            bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
            bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
            chunk_max_vocab_size=self.hparams['chunk_max_vocab_size'],
            chunk_freq_threshold=self.hparams['chunk_freq_threshold'],
            min_chunk_len=self.hparams['min_chunk_len'],
            max_chunk_len=self.hparams['max_chunk_len'],
            add_gold_chunk=self.hparams['use_gold_chunk'],
            add_nongold_chunk=not self.hparams['unuse_nongold_chunk'],
            add_unknown_pretrained_chunk=self.hparams['use_unknown_pretrained_chunk'],
            unigram_vocab=(self.unigram_embed_model.wv if self.unigram_embed_model else set()),
            bigram_vocab=(self.bigram_embed_model.wv if self.bigram_embed_model else set()),
            chunk_vocab=(self.chunk_embed_model.wv if self.chunk_embed_model else set()),
            generate_ngram_chunks=self.args.gen_oov_chunk_for_test,
            trie_ext=self.dic_ext,
        )


    def load_data(self, data_type):
        super().load_data(data_type)
        if data_type == 'train':
            data = self.train
            dic = self.dic
            evaluate = True
        elif data_type == 'devel':
            data = self.dev
            dic = self.dic_dev
            evaluate = True
        elif data_type == 'test':
            data = self.test
            dic = self.dic
            evaluate = True
        elif data_type == 'decode':
            data = self.decode_data
            dic = self.dic
            evaluate = False

        self.log('Start chunk search for {} data (min_len={}, max_len={})\n'.format(
            data_type, self.hparams['min_chunk_len'], self.hparams['max_chunk_len']))
        segmentation_data_loader.add_chunk_sequences(
            data, dic, 
            min_len=self.hparams['min_chunk_len'], 
            max_len=self.hparams['max_chunk_len'], 
            evaluate=evaluate, model_type=self.hparams['chunk_pooling_type'])


    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_dim = hparams['unigram_embed_dim']
        chunk_embed_dim = hparams['chunk_embed_dim']
        n_chunks = len(dic.tries[constants.CHUNK]) if dic.has_trie(constants.CHUNK) else 1
     
        if 'bigram_embed_dim' in hparams and hparams['bigram_embed_dim'] > 0:
            bigram_embed_dim = hparams['bigram_embed_dim']
            n_bigrams = len(dic.tables[constants.BIGRAM])
        else:
            bigram_embed_dim = n_bigrams = 0
     
        if 'pretrained_unigram_embed_dim' in hparams and hparams['pretrained_unigram_embed_dim'] > 0:
            pretrained_unigram_embed_dim = hparams['pretrained_unigram_embed_dim']
        else:
            pretrained_unigram_embed_dim = 0
     
        if 'pretrained_bigram_embed_dim' in hparams and hparams['pretrained_bigram_embed_dim'] > 0:
            pretrained_bigram_embed_dim = hparams['pretrained_bigram_embed_dim']
        else:
            pretrained_bigram_embed_dim = 0
     
        if 'pretrained_chunk_embed_dim' in hparams and hparams['pretrained_chunk_embed_dim'] > 0:
            pretrained_chunk_embed_dim = hparams['pretrained_chunk_embed_dim']
        else:
            pretrained_chunk_embed_dim = 0
     
        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE
     
        n_label = len(dic.tables[constants.SEG_LABEL])
        n_labels = [n_label]
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
     
            if pretrained_chunk_embed_dim > 0 and pretrained_chunk_embed_dim != chunk_embed_dim:
                print('Error: pre-trained and random initialized chunk embedding vectors '
                      + 'must be the same dimension for {} operation'.format(hparams['pretrained_embed_usage'])
                      + ': d1={}, d2={}'.format(pretrained_chunk_embed_dim, chunk_embed_dim),
                      file=sys.stderr)
                sys.exit()
        
        predictor = models.tagger.construct_RNNTagger(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim,
            n_attr1, attr1_embed_dim, n_chunks, chunk_embed_dim, 
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], 
            hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['rnn_n_layers2'] if 'rnn_n_layers2' in hparams else 0,
            hparams['rnn_n_units2'] if 'rnn_n_units2' in hparams else 0,
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels[0], 
            use_crf=hparams['inference_layer'] == 'crf',
            feat_dim=hparams['additional_feat_dim'], mlp_n_additional_units=0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout'] if 'biaffine_dropout' in hparams else 0.0,
            embed_dropout=hparams['embed_dropout'] if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            chunk_vector_dropout=hparams['chunk_vector_dropout'] if 'chunk_vector_dropout' in hparams else 0.0,
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim,
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage,
            chunk_pooling_type=hparams['chunk_pooling_type'] if 'chunk_pooling_type' in hparams else '',
            min_chunk_len=hparams['min_chunk_len'] if 'min_chunk_len' in hparams else 0,
            max_chunk_len=hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0,
            chunk_loss_ratio=hparams['chunk_loss_ratio'] if 'chunk_loss_ratio' in hparams else 0.0,
            biaffine_type=hparams['biaffine_type'] if 'biaffine_type' in hparams else '')

        self.classifier = classifiers.hybrid_sequence_tagger.HybridSequenceTagger(predictor, task=self.task)


    def gen_vocabs(self):
        if not self.task == constants.TASK_SEG:
            return

        train_path = self.args.path_prefix + self.args.train_data
        char_table = self.dic.tables[constants.UNIGRAM]

        if self.dic_org:        # test
            dics = [self.dic_org, self.dic]
        elif self.dic_dev:      # dev
            dics = [self.dic, self.dic_dev]
        else:                   # train
            dics = [self.dic]

        vocabs = self.data_loader.gen_vocabs(
            train_path, char_table, *dics, data_format=self.args.input_data_format)

        return vocabs


    def setup_evaluator(self, evaluator=None):
        # TODO ignored_labels
        evaluator1 = None
        if self.task == constants.TASK_SEG:
            if self.args.evaluation_method == 'normal':
                evaluator1 = HybridSegmenterEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)
            elif self.args.evaluation_method == 'each_length':
                evaluator1 = FMeasureEvaluatorForEachLength(
                    self.dic.tables[constants.SEG_LABEL].id2str)
            elif self.args.evaluation_method == 'each_vocab':
                vocabs = self.gen_vocabs()
                evaluator1 = FMeasureEvaluatorForEachVocab(
                    self.dic.tables[constants.SEG_LABEL].id2str, vocabs)
            elif self.args.evaluation_method == 'attention':
                evaluator1 = AccuracyEvaluatorForAttention()            

        elif self.task == constants.TASK_SEGTAG:
            evaluator1 = HybridTaggerEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)

        if not evaluator:
            self.evaluator = evaluator1
        else:
            evaluator = evaluator1


    def gen_inputs(self, data, ids, evaluate=True, restrict_memory=False):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        # for error analysis
        # index = ids[0]
        # id2token = self.dic.tables[constants.UNIGRAM].id2str
        # id2chunk = self.dic.tries[constants.CHUNK].id2chunk,
        # id2label = self.dic.tables[constants.SEG_LABEL].id2str
        # print(index)
        # print('l', [str(i)+':'+id2label[int(e)] for i,e in enumerate(data.outputs[0][index])])
        # print('u', [str(i)+':'+id2token[int(e)] for i,e in enumerate(data.inputs[0][index])])
        # print('c')
        # for i in range(len(data.inputs[0][index])):
        #     print(' ', i, [str(k)+':'+id2chunk[0][data.inputs[5][index][k][i]] for k in range(10)])
        # print('c', [str(i)+':'+id2chunk[int(e)] for i,e in enumerate(data.inputs[4][index])])

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        cs = [xp.asarray(data.inputs[3][j], dtype='i') for j in ids] if data.inputs[3] else None
        ds = [xp.asarray(data.inputs[4][j], dtype='i') for j in ids] if data.inputs[4] else None

        if (self.hparams['chunk_pooling_type'] == constants.CON or
            self.hparams['chunk_pooling_type'] == constants.WCON):
            feat_size = sum([h for h in range(
                self.hparams['min_chunk_len'], self.hparams['max_chunk_len']+1)])
            emb_dim = self.hparams['chunk_embed_dim']
        else:
            feat_size = 0
            emb_dim = 0
            # ms = [data.inputs[6][j] for j in ids]

        # for i, j in enumerate(ids):
        #     print('{}-th example (id={})'.format(i, j))
        #     print(len(us[i]), us[i])
        #     print(len(cs[i]), cs[i])
        #     print('mask', data.inputs[6][j])
        #     print()

        use_attention = (self.hparams['chunk_pooling_type'] == constants.WAVG or
                         self.hparams['chunk_pooling_type'] == constants.WCON)
        ms = [segmentation_data_loader.convert_mask_matrix(
            data.inputs[5][j], len(us[i]), len(cs[i]) if cs else 0, feat_size, emb_dim,
            use_attention, xp=xp) for i, j in enumerate(ids)]
        # print('u', us[0].shape, us[0])
        # print('c', cs[0].shape if cs is not None else None, cs[0] if cs is not None else None)
        # print('d', ds[0].shape if ds is not None else None, ds[0] if ds is not None else None)
        # print('m0', ms[0][0].shape if ms[0][0] is not None else None, ms[0][0])
        # print('m1', ms[0][1].shape if ms[0][1] is not None else None, ms[0][1])
        # print('m2', ms[0][2].shape if ms[0][2] is not None else None, ms[0][2])

        if self.hparams['feature_template']:
            if self.args.batch_feature_extraction:
                fs = [data.featvecs[j] for j in ids]
            else:
                us_int = [data.inputs[0][j] for j in ids]
                fs = self.feat_extractor.extract_features(us_int, self.dic.tries[constants.CHUNK])
        else:
            fs = None                

        gls = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None
        gcs = [xp.asarray(data.outputs[1][j], dtype='i') for j in ids] if evaluate else None


        if evaluate:
            if self.args.evaluation_method == 'attention':
                ncands = [get_num_candidates(data.inputs[5][j][0], len(us[i])) for i, j in enumerate(ids)]
                return us, cs, ds, ms, bs, fs, gls, gcs, ncands

            else:
                return us, cs, ds, ms, bs, fs, gls, gcs
        else:
            return us, cs, ds, ms, bs, fs


    def get_labels(self, inputs):
        if (self.args.evaluation_method == 'each_length' or
            self.args.evaluation_method == 'each_vocab'):
            return inputs[self.label_begin_index:self.label_begin_index+1]
        else:
            return inputs[self.label_begin_index:]


    def get_outputs(self, ret):
        if (self.args.evaluation_method == 'each_length' or
            self.args.evaluation_method == 'each_vocab'):
            return ret[1:2]
        else:
            return ret[1:]
        

    # tmp
    def load_external_dictionary(self):
        if self.args.external_dic_path:
            edic_path = self.args.external_dic_path
            attr_delim = constants.WL_ATTR_DELIM

            trie_ext = dictionary.MapTrie()
            with open(edic_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(constants.COMMENT_SYM):
                        continue
         
                    arr = line.split(attr_delim)
                    if len(arr[0]) == 0:
                        continue
         
                    word = arr[0]
                    trie_ext.get_chunk_id(word, word, update=True)

            self.dic_ext = trie_ext
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of chunks: {}'.format(len(trie_ext)))
            self.log('')
