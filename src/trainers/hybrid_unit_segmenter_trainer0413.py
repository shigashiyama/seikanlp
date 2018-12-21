import sys

import numpy as np

from chainer import cuda

import classifiers.hybrid_sequence_tagger
import common
import constants
from data import chunk_util0413
from models.tagger0413 import RNNTaggerWithChunk0413
import models.util
from trainers.trainer import Trainer
from trainers.tagger_trainer import TaggerTrainerBase
from trainers.hybrid_unit_segmenter_trainer import HybridUnitSegmenterTrainer


class HybridUnitSegmenterTrainer0413(HybridUnitSegmenterTrainer):
    def load_data(self, data_type):
        Trainer.load_data(self, data_type)
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

        xp = cuda.cupy if self.args.gpu >= 0 else np
        use_attention = 'W' in self.hparams['chunk_pooling_type']
        use_concat = 'CON' in self.hparams['chunk_pooling_type']
        chunk_util0413.add_chunk_sequences(
            data, dic, 
            max_len=self.hparams['max_chunk_len'], 
            evaluate=evaluate,
            use_attention=use_attention, use_concat=use_concat, xp=xp)

    
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

        ## from here: diff from latest 
        predictor = RNNTaggerWithChunk0413(
            n_vocab, unigram_embed_dim, n_bigrams, bigram_embed_dim, n_tokentypes, tokentype_embed_dim,
            n_attr1, attr1_embed_dim, n_chunks, chunk_embed_dim, 
            hparams['rnn_unit_type'], hparams['rnn_bidirection'], 
            hparams['rnn_n_layers'], hparams['rnn_n_units'], 
            hparams['rnn_n_layers2'] if 'rnn_n_layers2' in hparams else 0,
            hparams['rnn_n_units2'] if 'rnn_n_units2' in hparams else 0,
            hparams['mlp_n_layers'], hparams['mlp_n_units'], n_labels[0], 
            use_crf=hparams['inference_layer'] == 'crf',
            feat_dim=hparams['additional_feat_dim'], #mlp_n_additional_units=0,
            embed_dropout=hparams['embed_dropout'] if 'embed_dropout' in hparams else 0.0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout'] if 'biaffine_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            chunk_vector_dropout=hparams['chunk_vector_dropout'] if 'chunk_vector_dropout' in hparams else 0.0,
            pretrained_unigram_embed_dim=pretrained_unigram_embed_dim,
            pretrained_bigram_embed_dim=pretrained_bigram_embed_dim,
            pretrained_chunk_embed_dim=pretrained_chunk_embed_dim,
            pretrained_embed_usage=pretrained_embed_usage,
            chunk_pooling_type=hparams['chunk_pooling_type'] if 'chunk_pooling_type' in hparams else '',
            # min_chunk_len=hparams['min_chunk_len'] if 'min_chunk_len' in hparams else 0,
            max_chunk_len=hparams['max_chunk_len'] if 'max_chunk_len' in hparams else 0,
            chunk_loss_ratio=hparams['chunk_loss_ratio'] if 'chunk_loss_ratio' in hparams else 0.0,
            biaffine_type=hparams['biaffine_type'] if 'biaffine_type' in hparams else '')

        self.classifier = classifiers.hybrid_sequence_tagger.HybridSequenceTagger(predictor, task=self.task)


    def gen_inputs(self, data, ids, evaluate=True, restrict_memory=False): # from old src
        xp = cuda.cupy if self.args.gpu >= 0 else np

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        ts = [xp.asarray(data.inputs[2][j], dtype='i') for j in ids] if data.inputs[2] else None
        ss = None
        cs = [xp.asarray(data.inputs[4][j], dtype='i') for j in ids] if data.inputs[4] else None
        ds = [xp.asarray(data.inputs[5][j], dtype='i') for j in ids] if data.inputs[5] else None

        use_concat = 'CON' in self.hparams['chunk_pooling_type']
        use_attention = 'W' in self.hparams['chunk_pooling_type']
        if use_concat:
            feat_size = sum([h for h in range(self.hparams['max_chunk_len']+1)])
            emb_dim = self.hparams['chunk_embed_dim']
            ms = [chunk_util0413.convert_mask_matrix(
                data.inputs[6][j], len(us[i]), len(cs[i]) if cs else 0, feat_size, emb_dim,
                use_attention, use_concat, xp=xp) for i, j in enumerate(ids)]
        else:
            # feat_size = 0
            # emb_dim = 0
            ms = [data.inputs[6][j] for j in ids]

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
