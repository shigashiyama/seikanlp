import sys
from collections import Counter # tmp

import numpy as np

import chainer
from chainer import cuda

import classifiers.hybrid_sequence_tagger
import common
import constants
from data import segmentation_data_loader, tagging_data_loader
import evaluators
import models.tagger
import models.util
import optimizers
from trainers import trainer
from trainers.tagger_trainer import TaggerTrainerBase


# character word hybrid segmenter/tagger
class HybridUnitSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.chunk_embed_model = None
        self.label_begin_index = 8


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
                use_attention=('w' in self.hparams['chunk_pooling_type']),
                use_concat=('con' in self.hparams['chunk_pooling_type']))

            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


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
            finetuning = not self.hparams['fix_pretrained_embed']
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, self.bigram_embed_model, self.chunk_embed_model, finetuning)
        else:
            # called from super.load_model
            pass


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
                self.classifier.change_rnn_dropout_ratio(self.hparams['biaffine_dropout'])
            if 'chunk_vector_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(self.hparams['chunk_vector_dropout'])
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
                dic, self.unigram_embed_model, self.bigram_embed_model, self.chunk_embed_model,
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

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_chunk_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_bigram_embed_dim' : pretrained_bigram_embed_dim,
            'pretrained_chunk_embed_dim' : pretrained_chunk_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'fix_pretrained_embed' : self.args.fix_pretrained_embed,
            'chunk_loss_ratio' : self.args.chunk_loss_ratio,
            'chunk_pooling_type' : self.args.chunk_pooling_type,
            'biaffine_type' : self.args.biaffine_type,
            'use_gold_chunk' : self.args.use_gold_chunk,
            'use_unknown_pretrained_chunk' : not self.args.ignore_unknown_pretrained_chunk,
            'max_chunk_len' : self.args.max_chunk_len,
            'unigram_embed_dim' : self.args.unigram_embed_dim,
            'bigram_embed_dim' : self.args.bigram_embed_dim,
            'tokentype_embed_dim' : self.args.tokentype_embed_dim,
            'subtoken_embed_dim' : self.args.subtoken_embed_dim,
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
            'tagging_scheme': self.args.tagging_scheme,
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
                    key == 'max_chunk_len' or
                    key == 'unigram_embed_dim' or
                    key == 'bigram_embed_dim' or
                    key == 'tokentype_embed_dim' or
                    key == 'subtoken_embed_dim' or
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
                      key == 'fix_pretrained_embed' or
                      key == 'use_gold_chunk' or
                      key == 'ignore_unknown_pretrained_chunk'

                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        # tmp
        # if not 'tagging_scheme'in hparams:
        #     hparams['tagging_scheme'] = 'BIES'

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

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_chunk_embed_dim'] != pretrained_chunk_embed_dim:
                self.log(
                    'Error: pretrained_chunk_embed_dim and dimension of loaded embedding model'
                    + 'are conflicted.'.format(
                        hparams['pretrained_chunk_embed_dim'], pretrained_chunk_embed_dim))
                sys.exit()


    def setup_data_loader(self):
        # scheme = hparams['tagging_scheme']
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)

        self.data_loader = segmentation_data_loader.SegmentationDataLoader(
            token_index=self.args.token_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            use_bigram=(self.hparams['bigram_embed_dim'] > 0),
            use_tokentype=(self.hparams['tokentype_embed_dim'] > 0),
            use_chunk_trie=True,
            bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
            bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
            chunk_max_vocab_size=self.hparams['chunk_max_vocab_size'],
            chunk_freq_threshold=self.hparams['chunk_freq_threshold'],
            max_chunk_len=self.hparams['max_chunk_len'],
            add_gold_chunk=self.hparams['use_gold_chunk'],
            add_unknown_pretrained_chunk=self.hparams['use_unknown_pretrained_chunk'],
            unigram_vocab=(self.unigram_embed_model.wv if self.unigram_embed_model else set()),
            bigram_vocab=(self.bigram_embed_model.wv if self.bigram_embed_model else set()),
            chunk_vocab=(self.chunk_embed_model.wv if self.chunk_embed_model else set()),
        )


    def load_data(self, data_type):
        super().load_data(data_type)
        if self.hparams['feature_template']:
            self.log('Start feature extraction for {} data\n'.format(data_type))
            self.extract_features(data)

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

        self.log('Start chunk search for {} data (max_len={})\n'.format(
            data_type, self.hparams['max_chunk_len']))
        segmentation_data_loader.add_chunk_sequences(
            data, dic, 
            max_len=self.hparams['max_chunk_len'], evaluate=evaluate,
            use_attention=('w' in self.hparams['chunk_pooling_type']),
            use_concat=('con' in self.hparams['chunk_pooling_type']))


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []
        if self.unigram_embed_model and self.hparams['fix_pretrained_embed']:
            delparams.append('pretrained_unigram_embed/W')
        if self.bigram_embed_model and self.hparams['fix_pretrained_embed']:
            delparams.append('pretrained_bigram_embed/W')
        if self.chunk_embed_model and self.hparams['fix_pretrained_embed']:
            delparams.append('pretrained_chunk_embed/W')

        if delparams:
            for params in delparams:
                self.log('Fix parameters: {}'.foramt(param))                
            self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_dim = hparams['unigram_embed_dim']
        chunk_embed_dim = hparams['chunk_embed_dim']
        n_chunks = len(dic.tries[constants.CHUNK])
     
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
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
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
            max_chunk_len=hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0,
            chunk_loss_ratio=hparams['chunk_loss_ratio'] if 'chunk_loss_ratio' in hparams else 0.0,
            biaffine_type=hparams['biaffine_type'] if 'biaffine_type' in hparams else '')

        self.classifier = classifiers.hybrid_sequence_tagger.HybridSequenceTagger(predictor, task=self.task)


    def setup_evaluator(self):
        # TODO ignored_labels
        evaluator = None
        if self.task == constants.TASK_SEG:
            evaluator = evaluators.HybridSegmenterEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)

        elif self.task == constants.TASK_SEGTAG:
            evaluator = evaluators.HybridTaggerEvaluator(self.dic.tables[constants.SEG_LABEL].id2str)

        self.evaluator = evaluator


    def gen_inputs(self, data, ids, evaluate=True, restrict_memory=False):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        index = ids[0]
        id2token = self.dic.tables[constants.UNIGRAM].id2str
        id2chunk = self.dic.tries[constants.CHUNK].id2chunk,
        id2label = self.dic.tables[constants.SEG_LABEL].id2str

        # for error analysis
        # print(index)
        # print('l', [str(i)+':'+id2label[int(e)] for i,e in enumerate(data.outputs[0][index])])
        # print('u', [str(i)+':'+id2token[int(e)] for i,e in enumerate(data.inputs[0][index])])
        # print('c')
        # for i in range(len(data.inputs[0][index])):
        #     print(' ', i, [str(k)+':'+id2chunk[0][data.inputs[5][index][k][i]] for k in range(10)])
        # print('c', [str(i)+':'+id2chunk[int(e)] for i,e in enumerate(data.inputs[4][index])])

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        ts = [xp.asarray(data.inputs[2][j], dtype='i') for j in ids] if data.inputs[2] else None
        ss = None
        cs = [xp.asarray(data.inputs[4][j], dtype='i') for j in ids] if data.inputs[4] else None
        ds = [xp.asarray(data.inputs[5][j], dtype='i') for j in ids] if data.inputs[5] else None

        use_concat = 'con' in self.hparams['chunk_pooling_type']
        use_attention = 'w' in self.hparams['chunk_pooling_type']
        if use_concat:
            feat_size = sum([h for h in range(self.hparams['max_chunk_len']+1)])
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

        ms = [segmentation_data_loader.convert_mask_matrix(
            data.inputs[6][j], len(us[i]), len(cs[i]) if cs else 0, feat_size, emb_dim,
            use_attention, xp=xp) for i, j in enumerate(ids)]
            # use_attention, use_concat, xp=xp) for i, j in enumerate(ids)]

        # print('u', us[0].shape, us[0])
        # print('c', cs[0].shape if cs is not None else None, cs[0] if cs is not None else None)
        # print('d', ds[0].shape if ds is not None else None, ds[0] if ds is not None else None)
        # print('m0', ms[0][0].shape if ms[0][0] is not None else None, ms[0][0])
        # print('m1', ms[0][1].shape if ms[0][1] is not None else None, ms[0][1])
        # print('m2', ms[0][2].shape if ms[0][2] is not None else None, ms[0][2])

        fs = [data.featvecs[j] for j in ids] if self.hparams['feature_template'] else None
        gls = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None
        gcs = [xp.asarray(data.outputs[1][j], dtype='i') for j in ids] if evaluate else None

        if evaluate:
            return us, cs, ds, ms, bs, ts, ss, fs, gls, gcs
        else:
            return us, cs, ds, ms, bs, ts, ss, fs


    # tmp
    def run_train_mode2(self):
        for i in range(2):
            print('<train>' if i == 0 else '<devel>')
            data = self.train if i == 0 else self.dev
            counter = Counter()
            n_tokens = 0

            n_ins = len(data.inputs[0])
            ids = [i for i in range(n_ins)]
            inputs = self.gen_inputs(data, ids)
            ms = inputs[2]

            # count
            total_chunks = 0
            for m in ms:
                sen_len = m.shape[0]
                n_tokens += m.shape[0]
                for i in range(sen_len):
                    n_cands = sum(m[i])
                    counter[n_cands] += 1
                    total_chunks += 1

            ave = 0
            for cand, count in counter.items():
                rate = count / total_chunks
                ave += cand * rate
                print(cand, count, rate * 100)
            print('ave', ave)
            print()

    # tmp
    def run_eval_mode2(self):
        results = {'total': Counts(), 
                   'atn_cor' : Counts(),
                   'atn_inc' : Counts(),
                   'atn_inc_unk' : Counts()}

        # xp = cuda.cupy if self.args.gpu >= 0 else np
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)
        classifier.predictor.set_dics(
            self.dic.tables[constants.UNIGRAM].id2str,
            self.dic.tries[constants.CHUNK].id2chunk,
            self.dic.tables[constants.SEG_LABEL].id2str)
        
        data = self.test
        n_ins = len(data.inputs[0])
        n_sen = 0
        total_counts = None
                
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            n_sen += len(xs)
            golds = inputs[self.label_begin_index:]
            ncands = [get_num_candidates(data.inputs[6][j], len(xs[i])) for i, j in enumerate(ids)]

            with chainer.no_backprop_mode():
                gcs, pcs, gls, pls, _ = classifier.predictor.do_analysis(*inputs)

            counts = self.evaluator.calculate(*[xs], *golds, pls, pcs)
            total_counts = evaluators.merge_counts(total_counts, counts)

            if not gcs:
                gcs = [None] * len(xs)
            if not pcs:
                pcs = [None] * len(xs)

            for gc, pc, ncand, gl, pl in zip(gcs, pcs, ncands, gls, pls):
                if gc is None:
                    gc = [-1] * len(gl)
                if pc is None:
                    pc = [None] * len(gl)

                # print('(sen)')
                # print(gc)
                # print(pc)
                # print(ncand)
                # print(gl)
                # print(pl)

                for gci, pci, nci, gli, pli in zip(gc, pc, ncand, gl, pl):
                    if not nci in results:
                        results[nci] = Counts()

                    results[nci].increment(
                        possible=gci >= 0,
                        wcorrect=gci == pci,
                        scorrect=gli == pli)

                    results['total'].increment(
                        possible=gci >= 0,
                        wcorrect=gci == pci,
                        scorrect=gli == pli)

                    if pci is not None:
                        if pci == gci:
                            results['atn_cor'].increment(
                                possible=True,
                                wcorrect=True,
                                scorrect=gli == pli)

                        else:
                            results['atn_inc'].increment(
                                possible=gci >= 0,
                                wcorrect=False,
                                scorrect=gli == pli)

                            if gci < 0:
                                results['atn_inc_unk'].increment(
                                    possible=False,
                                    wcorrect=False,
                                    scorrect=gli == pli)

        print('Finished', n_sen)
        print_results(results)

        print('<results>')
        self.evaluator.report_results(n_sen, total_counts, 0, file=sys.stderr)


    def run_eval_mode3(self):
        # xp = cuda.cupy if self.args.gpu >= 0 else np
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)
        classifier.predictor.set_dics(
            self.dic.tables[constants.UNIGRAM].id2str,
            self.dic.tries[constants.CHUNK].id2chunk,
            self.dic.tables[constants.SEG_LABEL].id2str)
        
        id2token = self.dic.tables[constants.UNIGRAM].id2str
        id2chunk = self.dic.tries[constants.CHUNK].id2chunk
        id2label = self.dic.tables[constants.SEG_LABEL].id2str

        data = self.test
        n_ins = len(data.inputs[0])
        n_sen = 0
        total_counts = None
                
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            ws = inputs[1]
            golds = inputs[self.label_begin_index:]
            ncands = [get_num_candidates(data.inputs[6][j], len(xs[i])) for i, j in enumerate(ids)]

            with chainer.no_backprop_mode():
                gcs, pcs, gls, pls, ass = classifier.predictor.do_analysis(*inputs)

            if not gcs:
                gcs = [None] * len(xs)
            if not pcs:
                pcs = [None] * len(xs)

            for x, w, gc, pc, ncand, gl, pl, ascores in zip(xs, ws, gcs, pcs, ncands, gls, pls, ass):
                n = len(x)
                chunks = [id2chunk[int(wi)] for wi in w]

                if gc is None:
                    gc = [-1] * len(gl)
                if pc is None:
                    pc = [None] * len(gl)

                i = 0
                for xi, gli, pli, gci, pci, si in zip(x, gl, pl, gc, pc, ascores):
                    label_correct = 'L-T' if gli == pli else 'L-F'
                    att_correct = 'A-T' if gci == pci else 'A-F'
                    indexes = models.tagger.char_index2feat_indexes(i, n, self.hparams['max_chunk_len'])
                    gj = indexes[int(gci)]
                    pj = indexes[int(pci)]
                    gchunk = chunks[gj] if gj >= 0 else None
                    pchunk = chunks[pj]

                    print('{}-{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\t{}\t{}'.format(
                        n_sen, i, id2token[int(xi)], 
                        label_correct, id2label[int(gli)], id2label[int(pli)], 
                        att_correct, float(si), gchunk, pchunk)
                    )
                    i += 1
                print()
                n_sen += 1

        print('Finished', n_sen)


    def run_eval_mode4(self):
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)
        classifier.predictor.set_dics(
            self.dic.tables[constants.UNIGRAM].id2str,
            self.dic.tries[constants.CHUNK].id2chunk,
            self.dic.tables[constants.SEG_LABEL].id2str)
        
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
                _, _, gls, pls, __ = classifier.predictor.do_analysis(*inputs)

            for gl, pl in zip(gls, pls):
                # print('(sen)')
                # print(gl)
                # print(pl)

                for gli, pli in zip(gl, pl):
                    print('{}'.format(int(gli) == int(pli)))
                print()



class Counts(object):
    def __init__(self):
        self.all = 0
        self.possible = 0       # correct word exists in candidates
        self.wcorrect = 0       # for word prediction
        self.scorrect = 0       # for segmentation label


    def __str__(self):
        return '%d %d %d %d' % (self.all, self.possible, self.wcorrect, self.scorrect)


    def increment(self, possible=True, wcorrect=True, scorrect=True):
        self.all += 1 
        if possible:
            self.possible += 1
        if wcorrect:
            self.wcorrect += 1
        if scorrect:
            self.scorrect += 1


def get_num_candidates(mask, n_tokens):
    pairs1 = mask[1]
    ncand = [0] * n_tokens
    for i, j in pairs1:
        ncand[i] += 1
    return ncand


def print_results(results):
    print('ncand\tall\tw_pos\tw_cor\ts_cor\tw_upper\tw_acc\ts_acc')
    for ncand, counts in results.items():
        # print(ncand, counts)
        wacc = counts.wcorrect / counts.all * 100
        wupper = counts.possible / counts.all * 100
        sacc = counts.scorrect / counts.all * 100
        print('%s\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f' % (
            str(ncand), counts.all, counts.possible, counts.wcorrect, counts.scorrect,
            wupper, wacc, sacc))
    if 1 in results:
        print('lower(1): %.2f' % (results[1].possible / results[1].all * 100))
        print('lower(total): %.2f' % (results[1].possible / results['total'].all * 100))
    print()


