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

import evaluators

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

        attr_indexes = common.get_attribute_values(self.hparams['attr_indexes'])
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
        if 'attr_depths' in self.hparams:
            attr_depths = common.get_attribute_values(self.hparams['attr_depths'])
        else:
            attr_depths = [-1]

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
                use_subtoken=use_subtoken, subpos_depth=attr_depths[0])

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
            # print(' '.join(y_str)) # tmp

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
            'attr_indexes' : self.args.attribute_column_indexes,
            'attr_depths' : self.args.attribute_depths,
            'attr_predictions' : self.args.attribute_predictions,
            'attr_target_labelsets' : self.args.attribute_target_labelsets,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
            'token_freq_threshold' : self.args.token_freq_threshold,
            'token_max_vocab_size' : self.args.token_max_vocab_size,
            'bigram_freq_threshold' : self.args.bigram_freq_threshold,
            'bigram_max_vocab_size' : self.args.bigram_max_vocab_size,
            # 'chunk_freq_threshold' : self.args.chunk_freq_threshold,
            # 'chunk_max_vocab_size' : self.args.chunk_max_vocab_size,
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
                      key == 'lowercase' or
                      key == 'normalize_digits' or
                      key == 'fix_pretrained_embed'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        # tmp
        if not 'attr_indexes' in hparams:
            hparams['attr_indexes'] = ''
        if not 'attr_depths' in hparams:
            hparams['attr_depths'] = ''
        if not 'attr_target_labelsets' in hparams:
            hparams['attr_target_labelsets'] = ''

        self.hparams = hparams
        self.task = self.hparams['task']

        # if self.hparams['bigram_embed_dim'] == 0 and self.args.bigram_embed_dim > 0:
        #     add_hparams = self.init_additional_hyperparameters()
        #     self.seg_hparams.update(self.hparams)
        #     self.hparams.update(add_hparams)
        #     self.dic.create_table(constants.BIGRAM)


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

        if self.bigram_embed_model:
            pretrained_bigram_embed_dim = self.bigram_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_bigram_embed_dim'] != pretrained_bigram_embed_dim:
                self.log(
                    'Error: pretrained_bigram_embed_dim and dimension of loaded embedding model'
                    + 'are conflicted.'.format(
                        hparams['pretrained_bigram_embed_dim'], pretrained_bigram_embed_dim))
                sys.exit()

    def load_data_for_training(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        super().load_data_for_training(refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams)


    def load_test_data(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        super().load_test_data(refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams)


    def load_decode_data(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        super().load_decode_data(refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams)


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

        attr_predictions = common.get_attribute_boolvalues(self.args.attribute_predictions)
        self.evaluator = classifiers.init_evaluator(
            self.task, self.dic, attr_predictions, self.args.ignored_labels)
        self.evaluator.calculator.id2token = self.dic.tables[constants.UNIGRAM].id2str # tmp


    # tmp
    # def run_train_mode2(self):
    #     id2label = self.dic.tables[constants.SEG_LABEL].id2str
    #     uni2label_tr = None
    #     bi2label_tr = None

    #     for i in range(2):
    #         print('<train>' if i == 0 else '<devel>')
    #         data = self.train if i == 0 else self.dev
    #         labels = Counter()
    #         uni2label = models.attribute_annotator.Key2Counter()
    #         bi2label = models.attribute_annotator.Key2Counter()
    #         n_tokens = 0

    #         n_ins = len(data.inputs[0])
    #         ids = [i for i in range(n_ins)]
    #         inputs = self.gen_inputs(data, ids)
    #         us = inputs[0]
    #         bs = inputs[1]
    #         ls = inputs[self.label_begin_index]

    #         # count
    #         for u, b, l in zip(us, bs, ls):
    #             n_tokens += len(u)
    #             for ui, bi, li in zip(u, b, l):
    #                 labels[li] += 1
    #                 uni2label.add(ui, li)
    #                 bi2label.add(bi, li)

    #         # calc label distribution
    #         for lab, cnt in labels.items():
    #             rate = cnt / n_tokens
    #             print(id2label[lab], rate*100)

    #         # calc entropy for unigram
    #         u_unk = 0
    #         u_ent = 0
    #         for u, counter in uni2label.key2counter.items():
    #             total = sum([v for v in counter.values()])
    #             ent = 0
    #             for key, count in counter.items():
    #                 p = count / total
    #                 ent += - p * np.log2(p)
    #             u_ent += ent
    #             if u == 0:
    #                 u_unk = total
    #                 print('unigram entropy (unk)', ent)

    #         u_ent /= len(uni2label)
    #         print('unigram entropy', u_ent)
    #         print('unigram unk ratio', u_unk / n_tokens)

    #         # calc entropy for bigram
    #         b_unk = 0
    #         b_ent = 0
    #         for b, counter in bi2label.key2counter.items():
    #             total = sum([v for v in counter.values()])
    #             ent = 0
    #             for key, count in counter.items():
    #                 p = count / total
    #                 ent += - p * np.log2(p)
    #             b_ent += ent
    #             if b == 0:
    #                 b_unk = total
    #                 print('bigram entropy (unk)', ent)


    #         b_ent /= len(bi2label)
    #         print('bigram entropy', b_ent)
    #         print('bigram unk ratio', b_unk / n_tokens)

    #         # evaluate using unigram
    #         if i == 0:
    #             uni2label_tr = uni2label
    #             bi2label_tr = bi2label
    #             continue

    #         os = []
    #         for u in us:
    #             o = []
    #             for ui in u:
    #                 counter = uni2label_tr.get(ui)
    #                 if counter:
    #                     pred = counter.most_common(1)[0][0]
    #                 else:
    #                     pred = -1
    #                 o.append(pred)
    #             os.append(o)
                
    #         counts = self.evaluator.calculate(*[us], *[ls], *[os])
    #         res = self.evaluator.report_results(n_ins, counts, 0.0)

    #         # evaluate using bigram
    #         os = []
    #         for b in bs:
    #             o = []
    #             for bi in b:
    #                 counter = bi2label_tr.get(bi)
    #                 if counter:
    #                     pred = counter.most_common(1)[0][0]
    #                 else:
    #                     pred = -1
    #                 o.append(pred)
    #             os.append(o)
                
    #         counts = self.evaluator.calculate(*[bs], *[ls], *[os])
    #         res = self.evaluator.report_results(n_ins, counts, 0.0)

    #         print()


# character word hybrid segmenter
class HybridSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.chunk_embed_model = None
        self.label_begin_index = 8


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
            'attr_indexes' : self.args.attribute_column_indexes,
            'attr_depths' : self.args.attribute_depths,
            'attr_predictions' : self.args.attribute_predictions,
            'attr_target_labelsets' : self.args.attribute_target_labelsets,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
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
        if not 'attr_indexes' in hparams:
            hparams['attr_indexes'] = ''
        if not 'attr_depths' in hparams:
            hparams['attr_depths'] = ''
        if not 'attr_target_labelsets' in hparams:
            hparams['attr_target_labelsets'] = ''

        self.hparams = hparams
        self.task = constants.TASK_HSEG

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


    def load_data_for_training(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_data_for_training(
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks)

        xp = cuda.cupy if self.args.gpu >= 0 else np
        use_attention = 'w' in self.hparams['chunk_pooling_type']
        use_concat = 'con' in self.hparams['chunk_pooling_type']

        self.log('Start chunk search for training data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        data_io.add_chunk_sequences(
            self.train, self.dic, max_len=self.hparams['max_chunk_len'], evaluate=True,
            use_attention=use_attention, use_concat=use_concat)

        if self.dev:
            self.log('Start chunk search for development data (max_len={})\n'.format(
                self.hparams['max_chunk_len']))
            data_io.add_chunk_sequences(
                self.dev, self.dic_dev, max_len=self.hparams['max_chunk_len'], evaluate=True,
                use_attention=use_attention, use_concat=use_concat)


    def load_test_data(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_test_data(
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks)

        use_attention = 'w' in self.hparams['chunk_pooling_type']
        use_concat = 'con' in self.hparams['chunk_pooling_type']

        self.log('Start chunk search for test data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        data_io.add_chunk_sequences(
            self.test, self.dic, max_len=self.hparams['max_chunk_len'], evaluate=True,
            # use_attention=True, use_concat=use_concat) # tmp - 20180622 comment out
            use_attention=use_attention, use_concat=use_concat)


    def load_decode_data(self):
        refer_unigrams = self.unigram_embed_model.wv if self.unigram_embed_model else set()
        refer_bigrams = self.bigram_embed_model.wv if self.bigram_embed_model else set()
        refer_chunks = self.chunk_embed_model.wv if self.chunk_embed_model else set()
        super().load_decode_data(
            refer_unigrams=refer_unigrams, refer_bigrams=refer_bigrams, refer_chunks=refer_chunks)

        use_attention = 'w' in self.hparams['chunk_pooling_type']
        use_concat = 'con' in self.hparams['chunk_pooling_type']

        self.log('Start chunk search for decode data (max_len={})\n'.format(self.hparams['max_chunk_len']))
        xp = cuda.cupy if self.args.gpu >= 0 else np
        data_io.add_chunk_sequences(
            self.decode_data, self.dic, max_len=self.hparams['max_chunk_len'], evaluate=False,
            use_attention=use_attention, use_concat=use_concat)


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


    def setup_evaluator(self):
        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


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
        ms = [data_io.convert_mask_matrix(
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
                gcs, pcs, gls, pls = classifier.predictor.do_analysis(*inputs)

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


# class DualTaggerTrainer(TaggerTrainerBase):
#     def __init__(self, args, logger=sys.stderr):
#         super().__init__(args, logger)
#         self.su_tagger = None
#         self.label_begin_index = 2


#     def load_hyperparameters(self, hparams_path):
#         hparams = {}
#         with open(hparams_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line.startswith('#'):
#                     continue

#                 kv = line.split('=')
#                 key = kv[0]
#                 val = kv[1]

#                 if (key == 'pretrained_unigram_embed_dim' or
#                     key == 'unigram_embed_dim' or
#                     key == 'subtoken_embed_dim' or
#                     key == 'additional_feat_dim' or
#                     key == 'rnn_n_layers' or
#                     key == 'rnn_n_units' or
#                     key == 'mlp_n_layers'or
#                     key == 'mlp_n_units' or
#                     key == 'subpos_depth' or
#                     key == 'freq_threshold' or
#                     key == 'max_vocab_size'
#                 ):
#                     val = int(val)

#                 elif (key == 'rnn_dropout' or
#                       key == 'mlp_dropout'
#                 ):
#                     val = float(val)

#                 elif (key == 'rnn_bidirection' or
#                       key == 'lowercase' or
#                       key == 'normalize_digits' or
#                       key == 'fix_pretrained_embed'
#                 ):
#                     val = (val.lower() == 'true')

#                 hparams[key] = val

#         self.hparams = hparams

#         self.task = self.hparams['task']
        

#     def init_hyperparameters(self):
#         self.log('Initialize hyperparameters for full model\n')

#         # keep most of loaded params from sub model and update some params from args
#         self.log('### arguments')
#         for k, v in self.args.__dict__.items():
#             if ('dropout' in k or
#                 k == 'ignore_unknown_pretrained_chunk' or
#                 k == 'submodel_usage'):
#                 self.hparams[k] = v
#                 message = '{}={}'.format(k, v)            

#             elif k in self.hparams and v != self.hparams[k]:
#                 message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

#             else:
#                 message = '{}={}'.format(k, v)

#             self.log('# {}'.format(message))
#             self.report('[INFO] arg: {}'.format(message))
#         self.log('')


#     def init_model(self):
#         self.log('Initialize model from hyperparameters\n')
#         self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


#     def load_model(self):
#         if self.args.model_path:
#             super().load_model()
#             if 'rnn_dropout' in self.hparams:
#                 self.classifier.change_rnn_dropout_ratio(self.hparams['rnn_dropout'])
#             if 'hidden_mlp_dropout' in self.hparams:
#                 self.classifier.change_hidden_mlp_dropout_ratio(self.hparams['hidden_mlp_dropout'])

#         elif self.args.submodel_path:
#             self.load_submodel()
#             self.init_hyperparameters()

#         else:
#             print('Error: model_path or sub_model_path must be specified for {}'.format(self.task))
#             sys.exit()


#     def load_submodel(self):
#         model_path = self.args.submodel_path
#         array = model_path.split('_e')
#         dic_path = '{}.s2i'.format(array[0])
#         hparam_path = '{}.hyp'.format(array[0])
#         param_path = model_path

#         # dictionary
#         self.load_dic(dic_path)

#         # hyper parameters
#         self.load_hyperparameters(hparam_path)
#         self.log('Load hyperparameters from short unit model: {}\n'.format(hparam_path))
#         for k, v in self.args.__dict__.items():
#             if 'dropout' in k:
#                 self.hparams[k] = 0.0
#             elif k == 'task':
#                 task = 'dual_{}'.format(self.hparams[k])
#                 self.task = self.hparams[k] = task

#         # sub model
#         hparams_sub = self.hparams.copy()
#         self.log('Initialize short unit model from hyperparameters\n')
#         task_tmp = self.task.split('dual_')[1]
#         classifier_tmp = classifiers.init_classifier(
#             task_tmp, hparams_sub, self.dic)
#         chainer.serializers.load_npz(model_path, classifier_tmp)
#         self.su_tagger = classifier_tmp.predictor
#         self.log('Load short unit model parameters: {}\n'.format(model_path))


#     def update_model(self):
#         if (self.args.execute_mode == 'train' or
#             self.args.execute_mode == 'eval' or
#             self.args.execute_mode == 'decode'):

#             super().update_model()
#             if self.su_tagger is not None:
#                 self.classifier.integrate_submodel(self.su_tagger)


#     def setup_optimizer(self):
#         super().setup_optimizer()

#         delparams = []
#         delparams.append('su_tagger')
#         self.log('Fix parameters: su_tagger/*')
#         self.optimizer.add_hook(optimizers.DeleteGradient(delparams))


#     def setup_evaluator(self):
#         if self.task == constants.TASK_DUAL_TAG and self.args.ignored_labels:
#             tmp = self.args.ignored_labels
#             self.args.ignored_labels = set()

#             for label in tmp.split(','):
#                 label_id = self.dic.tables[constants.POS_LABEL].get_id(label)
#                 if label_id >= 0:
#                     self.args.ignored_labels.add(label_id)

#             self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))

#         else:
#             self.args.ignored_labels = set()

#         self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


#     def decode_batch(self, *inputs, orgseqs, file=sys.stdout):
#         su_sym = 'S\t' if self.args.output_data_format == constants.WL_FORMAT else ''
#         lu_sym = 'L\t' if self.args.output_data_format == constants.WL_FORMAT else ''

#         ys_short, ys_long = self.classifier.decode(*inputs)

#         id2token = self.dic.tables[constants.UNIGRAM].id2str
#         id2label = self.dic.tables[constants.SEG_LABEL if 'seg' in self.task else constants.POS_LABEL].id2str

#         for x_str, sy, ly in zip(orgseqs, ys_short, ys_long):
#             sy_str = [id2label[int(yi)] for yi in sy]
#             ly_str = [id2label[int(yi)] for yi in ly]

#             if self.task == constants.TASK_DUAL_TAG:
#                 res1 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
#                         for xi_str, yi_str in zip(x_str, sy_str)]
#                 res1 = ' '.join(res1)
#                 res2 = ['{}{}{}'.format(xi_str, self.args.output_attr_delim, yi_str) 
#                         for xi_str, yi_str in zip(x_str, ly_str)]
#                 res2 = ' '.join(res2)

#             elif self.task == constants.TASK_DUAL_SEG:
#                 res1 = [xi_str + self.args.output_token_delim + su_sym
#                          if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
#                         for xi_str, yi_str in zip(x_str, ly_str)]
#                 res1 = ''.join(res1).rstrip('S\t\n')
#                 res2 = [xi_str + self.args.output_token_delim + lu_sym
#                         if yi_str.startswith('E') or yi_str.startswith('S') else xi_str
#                         for xi_str, yi_str in zip(x_str, sy_str)]
#                 res2 = ''.join(res2).rstrip('L\t\n')

#             elif self.task == constants.TASK_DUAL_SEGTAG:
#                 res1 = ['{}{}'.format(
#                     xi_str, 
#                     (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
#                     if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
#                 ) for xi_str, yi_str in zip(x_str, sy_str)]
#                 res1 = ''.join(res1).rstrip()
#                 res2 = ['{}{}'.format(
#                     xi_str, 
#                     (self.args.output_attr_delim+yi_str[2:]+self.args.output_token_delim) 
#                     if (yi_str.startswith('E-') or yi_str.startswith('S-')) else ''
#                 ) for xi_str, yi_str in zip(x_str, ly_str)]
#                 res2 = ''.join(res2).rstrip()

#             else:
#                 print('Error: Invalid decode type', file=self.logger)
#                 sys.exit()

#             if self.args.output_data_format == 'sl':
#                 print('S:{}'.format(res1), file=file)
#                 print('L:{}'.format(res2), file=file)
#             else:
#                 print('S\t{}'.format(res1), file=file)
#                 print('L\t{}'.format(res2), file=file)
#                 print(file=file)

