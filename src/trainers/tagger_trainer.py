import sys
import pickle

from collections import Counter # tmp
import numpy as np              # tmp

import chainer
from chainer import cuda

import common
import util
import constants
import data_io
import classifiers
import optimizers
from trainers import trainer
from trainers.trainer import Trainer


class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


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
            self.log('# seg labels: {}\n'.format(id2seg))
        if self.dic.has_table(constants.POS_LABEL):
            id2pos = {v:k for k,v in self.dic.tables[constants.POS_LABEL].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        ts = [xp.asarray(data.inputs[2][j], dtype='i') for j in ids] if data.inputs[2] else None
        ss = ([[xp.asarray(subs, dtype='i') for subs in data.inputs[1][j]] for j in ids] 
              if data.inputs[3] else None)
        fs = [data.featvecs[j] for j in ids] if self.hparams['feature_template'] else None
        ls = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None

        if evaluate:
            return us, bs, ts, ss, fs, ls
        else:
            return us, bs, ts, ss, fs
        

    def decode(self, rdata, use_pos=False, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        orgseqs = rdata.orgdata[0]

        timer = util.Timer()
        timer.start()
        for ids in trainer.batch_generator(n_ins, batch_size=self.args.batch_size, shuffle=False):
            inputs = self.gen_inputs(rdata, ids, evaluate=False)
            os = [orgseqs[j] for j in ids]
            self.decode_batch(*inputs, orgseqs=os, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def run_interactive_mode(self):
        use_pos = False
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
                line, self.dic, self.task, use_pos=use_pos,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=use_subtoken, subpos_depth=subpos_depth)

            if self.hparams['feature_template']:
                rdata.featvecs = self.feat_extractor.extract_features(ws[0], self.dic.tries[constants.CHUNK])

            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            orgseqs = rdata.orgdata[0]

            self.decode_batch(*inputs, orgseqs=orgseqs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)


    def decode_batch(self, *inputs, orgseqs, file=sys.stdout):
        ys = self.classifier.decode(*inputs)
        id2label = (self.dic.tables[constants.SEG_LABEL if common.is_segmentation_task(self.task) 
                                    else constants.POS_LABEL].id2str)

        for x_str, y in zip(orgseqs, ys):
            y_str = [id2label[int(yi)] for yi in y]
            print(' '.join(y_str)) # tmp

            if self.task == constants.TASK_TAG:
                res = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                       for xi_str, yi_str in zip(x_str, y_str)]
                res = self.args.output_token_delim.join(res)

            elif self.task == constants.TASK_SEG or constants.TASK_HSEG:
                res = ['{}{}'.format(xi_str, self.args.output_token_delim 
                                     if (yi_str.startswith('E') or yi_str.startswith('S')) 
                                     else '') for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

            elif self.task == constants.TASK_SEGTAG:
                res = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, y_str)]
                res = ''.join(res).rstrip()

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
            self.dic = data_io.load_external_dictionary(edic_path, use_pos=use_pos)
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
        self.label_begin_index = 5


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


    def update_model(self):
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
            'bigram_embed_dim' : self.args.bigram_embed_dim,
            'tokentype_embed_dim' : self.args.tokentype_embed_dim,
            'subtoken_embed_dim' : self.args.subtoken_embed_dim,
            'rnn_unit_type' : self.args.rnn_unit_type,
            'rnn_bidirection' : self.args.rnn_bidirection,
            'rnn_n_layers' : self.args.rnn_n_layers,
            'rnn_n_units' : self.args.rnn_n_units,
            'mlp_n_layers' : self.args.mlp_n_layers,
            'mlp_n_units' : self.args.mlp_n_units,
            'inference_layer' : self.args.inference_layer,
            'rnn_dropout' : self.args.rnn_dropout,
            'mlp_dropout' : self.args.mlp_dropout,
            'feature_template' : self.args.feature_template,
            'task' : self.args.task,
            'subpos_depth' : self.args.subpos_depth,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
            'freq_threshold' : self.args.freq_threshold,
            'max_vocab_size' : self.args.max_vocab_size,
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
                    key == 'bigram_embed_dim' or
                    key == 'tokentype_embed_dim' or
                    key == 'subtoken_embed_dim' or
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp_n_layers'or
                    key == 'mlp_n_units' or
                    key == 'subpos_depth' or
                    key == 'freq_threshold' or
                    key == 'max_vocab_size'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'mlp_dropout'
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
        

    def load_hyperparameter(self):
        super().load_hyperparameter(self)

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


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []
        if self.unigram_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_unigram_embed/W')
            self.log('Fix parameters: pretrained_unigram_embed/W')
        if delparams:
            self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_evaluator(self):
        if self.task == constants.TASK_TAG and self.args.ignored_labels:
            tmp = self.args.ignored_labels
            self.args.ignored_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.tables[constants.POS_LABEL].get_id(label)
                if label_id >= 0:
                    self.args.ignored_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))

        else:
            self.args.ignored_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)
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


# character word hybrid segmenter
class HybridSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.chunk_embed_model = None
        self.label_begin_index = 7


    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(self.args.unigram_embed_model_path)
        if self.args.chunk_embed_model_path:
            self.chunk_embed_model = trainer.load_embedding_model(self.args.chunk_embed_model_path)


    def init_model(self):
        super().init_model()
        # id2uni = self.dic.tables[constants.UNIGRAM].id2str
        # model = self.classifier.predictor
        # print('PU', model.pretrained_unigram_embed.W)
        # print('U', model.unigram_embed.W)

        finetuning = not self.hparams['fix_pretrained_embed']
        self.classifier.load_pretrained_embedding_layer(
            self.dic, self.unigram_embed_model, self.chunk_embed_model, finetuning)

        # print('PU', model.pretrained_unigram_embed.W)
        # print('U', model.unigram_embed.W)


    def load_model(self):
        super().load_model()
        if 'rnn_dropout' in self.hparams:
            self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
        if 'hidden_mlp_dropout' in self.hparams:
            self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])


    def update_model(self):
        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            self.classifier.grow_embedding_layers(
                self.dic, self.unigram_embed_model, self.chunk_embed_model,
                train=(self.args.execute_mode=='train'))
            self.classifier.grow_inference_layers(self.dic)


    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_dim = self.unigram_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_unigram_embed_dim = 0

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_chunk_embed_dim = 0

        self.hparams = {
            'pretrained_unigram_embed_dim' : pretrained_unigram_embed_dim,
            'pretrained_chunk_embed_dim' : pretrained_chunk_embed_dim,
            'pretrained_embed_usage' : self.args.pretrained_embed_usage,
            'fix_pretrained_embed' : self.args.fix_pretrained_embed,
            'use_attention' : self.args.use_attention,
            'use_chunk_first' : self.args.use_chunk_first,
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
            'rnn_dropout' : self.args.rnn_dropout,
            'biaffine_dropout' : self.args.biaffine_dropout,
            'mlp_dropout' : self.args.mlp_dropout,
            'feature_template' : self.args.feature_template,
            'task' : self.args.task,
            'subpos_depth' : self.args.subpos_depth,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
            'freq_threshold' : self.args.freq_threshold,
            'max_vocab_size' : self.args.max_vocab_size,
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
                    key == 'subpos_depth' or
                    key == 'freq_threshold' or
                    key == 'max_vocab_size'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'biaffine_dropout' or
                      key == 'mlp_dropout'
                ):
                    val = float(val)

                elif (key == 'rnn_bidirection' or
                      key == 'lowercase' or
                      key == 'normalize_digits' or
                      key == 'fix_pretrained_embed' or
                      key == 'use_attention' or
                      key == 'use_chunk_first'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams

        self.task = self.hparams['task']

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

        if self.chunk_embed_model:
            pretrained_chunk_embed_dim = self.chunk_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_chunk_embed_dim'] != pretrained_chunk_embed_dim:
                self.log(
                    'Error: pretrained_chunk_embed_dim and dimension of loaded embedding model'
                    + 'are conflicted.'.format(
                        hparams['pretrained_chunk_embed_dim'], pretrained_chunk_embed_dim))
                sys.exit()


    def load_data_for_training(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_data_for_training(refer_tokens=refer_tokens, refer_chunks=refer_chunks)
        
        # TODO confirm
        self.log('Start chunk search for training data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        xp = cuda.cupy if self.args.gpu >= 0 else np
        data_io.add_chunk_sequences(self.train, self.dic, max_len=self.args.max_chunk_len, xp=xp)

        if self.dev:
            self.log('Start chunk search for development data (max_len={})\n'.format(
                self.hparams['max_chunk_len']))
            data_io.add_chunk_sequences(self.dev, self.dic, max_len=self.args.max_chunk_len, xp=xp)


    def load_test_data(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_test_data(refer_tokens=refer_tokens, refer_chunks=refer_chunks)

        # TODO confirm: org - self.args.max_chunk_len
        self.log('Start chunk search for test data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        xp = cuda.cupy if self.args.gpu >= 0 else np
        data_io.add_chunk_sequences(self.test, self.dic, max_len=self.hparams['max_chunk_len'], xp=xp)


    def load_decode_data(self):
        refer_tokens = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_decode_data(refer_tokens=refer_tokens, refer_chunks=refer_chunks)

        self.log('Start chunk search for decode data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        xp = cuda.cupy if self.args.gpu >= 0 else np
        data_io.add_chunk_sequences(self.decode_data, self.dic, max_len=self.hparams['max_chunk_len'], xp=xp)


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []
        if self.unigram_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_unigram_embed/W')
            self.log('Fix parameters: pretrained_unigram_embed/W')

        if self.chunk_embed_model and self.args.fix_pretrained_embed:
            delparams.append('pretrained_chunk_embed/W')
            self.log('Fix parameters: pretrained_chunk_embed/W')

        if delparams:
            self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_evaluator(self):
        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


    def gen_inputs(self, data, ids, evaluate=True):
        xp = cuda.cupy if self.args.gpu >= 0 else np

        us = [xp.asarray(data.inputs[0][j], dtype='i') for j in ids]
        bs = [xp.asarray(data.inputs[1][j], dtype='i') for j in ids] if data.inputs[1] else None
        ts = [xp.asarray(data.inputs[2][j], dtype='i') for j in ids] if data.inputs[2] else None
        ss = None
        cs = [xp.asarray(data.inputs[4][j], dtype='i') for j in ids]
        ms = [data.inputs[5][j] for j in ids]
        fs = [data.featvecs[j] for j in ids] if self.hparams['feature_template'] else None
        ls = [xp.asarray(data.outputs[0][j], dtype='i') for j in ids] if evaluate else None

        # for u, c, m in zip(us, cs, ms):
        #     print(u)
        #     print([self.dic.tables[constants.UNIGRAM].id2str[int(i)] for i in u])

        #     print(c)
        #     print([self.dic.tries[constants.CHUNK].id2chunk[int(i)] for i in c])

        #     print(m)
        #     print()

        if evaluate:
            return us, cs, ms, bs, ts, ss, fs, ls
        else:
            return us, cs, ms, bs, ts, ss, fs


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


class DualTaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.su_tagger = None
        self.label_begin_index = 2


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
                    key == 'additional_feat_dim' or
                    key == 'rnn_n_layers' or
                    key == 'rnn_n_units' or
                    key == 'mlp_n_layers'or
                    key == 'mlp_n_units' or
                    key == 'subpos_depth' or
                    key == 'freq_threshold' or
                    key == 'max_vocab_size'
                ):
                    val = int(val)

                elif (key == 'rnn_dropout' or
                      key == 'mlp_dropout'
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
        

    def init_hyperparameters(self):
        self.log('Initialize hyperparameters for full model\n')

        # keep most of loaded params from sub model and update some params from args
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:  # update
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k == 'submodel_usage':
                self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


    def load_model(self):
        if self.args.model_path:
            super().load_model()
            if 'rnn_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
            if 'hidden_mlp_dropout' in self.hparams:
                self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])

        elif self.args.submodel_path:
            self.load_submodel()
            self.init_hyperparameters()

        else:
            print('Error: model_path or sub_model_path must be specified for {}'.format(self.task))
            sys.exit()


    def load_submodel(self):
        model_path = self.args.submodel_path
        array = model_path.split('_e')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters from short unit model: {}\n'.format(hparam_path))
        for k, v in self.args.__dict__.items():
            if 'dropout' in k:
                self.hparams[k] = 0.0
            elif k == 'task':
                task = 'dual_{}'.format(self.hparams[k])
                self.task = self.hparams[k] = task

        # sub model
        hparams_sub = self.hparams.copy()
        self.log('Initialize short unit model from hyperparameters\n')
        task_tmp = self.task.split('dual_')[1]
        classifier_tmp = classifiers.init_classifier(
            task_tmp, hparams_sub, self.dic)
        chainer.serializers.load_npz(model_path, classifier_tmp)
        self.su_tagger = classifier_tmp.predictor
        self.log('Load short unit model parameters: {}\n'.format(model_path))


    def update_model(self):
        if (self.args.execute_mode == 'train' or
            self.args.execute_mode == 'eval' or
            self.args.execute_mode == 'decode'):

            super().update_model()
            if self.su_tagger is not None:
                self.classifier.integrate_submodel(self.su_tagger)


    def setup_optimizer(self):
        super().setup_optimizer()

        delparams = []
        delparams.append('su_tagger')
        self.log('Fix parameters: su_tagger/*')
        self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


    def setup_evaluator(self):
        if self.task == constants.TASK_DUAL_TAG and self.args.ignored_labels:
            tmp = self.args.ignored_labels
            self.args.ignored_labels = set()

            for label in tmp.split(','):
                label_id = self.dic.tables[constants.POS_LABEL].get_id(label)
                if label_id >= 0:
                    self.args.ignored_labels.add(label_id)

            self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))

        else:
            self.args.ignored_labels = set()

        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


    def decode_batch(self, *inputs, orgseqs, file=sys.stdout):
        su_sym = 'S\t' if self.args.output_data_format == constants.WL_FORMAT else ''
        lu_sym = 'L\t' if self.args.output_data_format == constants.WL_FORMAT else ''

        ys_short, ys_long = self.classifier.decode(*inputs)

        id2token = self.dic.tables[constants.UNIGRAM].id2str
        id2label = self.dic.tables[constants.SEG_LABEL if 'seg' in self.task else constants.POS_LABEL].id2str

        for x_str, sy, ly in zip(orgseqs, ys_short, ys_long):
            sy_str = [id2label[int(yi)] for yi in sy]
            ly_str = [id2label[int(yi)] for yi in ly]

            if self.task == constants.TASK_DUAL_TAG:
                res1 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                        for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ' '.join(res1)
                res2 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
                        for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ' '.join(res2)

            elif self.task == constants.TASK_DUAL_SEG:
                res1 = [xi_str + self.args.output_token_delim + su_sym
                         if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
                        for xi_str, yi_str in zip(x_str, ly_str)]
                res1 = ''.join(res1).rstrip('S\t\n')
                res2 = [xi_str + self.args.output_token_delim + lu_sym
                        if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
                        for xi_str, yi_str in zip(x_str, sy_str)]
                res2 = ''.join(res2).rstrip('L\t\n')

            elif self.task == constants.TASK_DUAL_SEGTAG:
                res1 = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, sy_str)]
                res1 = ''.join(res1).rstrip()
                res2 = ['{}{}'.format(
                    xi_str, 
                    (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
                    if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
                ) for xi_str, yi_str in zip(x_str, ly_str)]
                res2 = ''.join(res2).rstrip()

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            if self.args.output_data_format == 'sl':
                print('S:{}'.format(res1), file=file)
                print('L:{}'.format(res2), file=file)
            else:
                print('S\t{}'.format(res1), file=file)
                print('L\t{}'.format(res2), file=file)
                print(file=file)
