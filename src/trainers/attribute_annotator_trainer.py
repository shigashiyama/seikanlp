from datetime import datetime
import pickle
import sys

import classifiers.pattern_matcher
import common
import constants
from data_loaders import data_loader, attribute_annotation_data_loader
from evaluators.common import AccuracyEvaluator
import models.attribute_annotator
from trainers import trainer
from trainers.trainer import Trainer
import util


class AttributeAnnotatorTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
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
        self.log('# train_gold_attr: {} ... {}\n'.format(train.outputs[0][0], train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        for i in range(len(attr_indexes)):
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                id2attr = {v:k for k,v in self.dic.tables[constants.ATTR_LABEL(i)].str2id.items()}
                self.log('# {}-th attribute labels: {}\n'.format(i, id2attr))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def init_hyperparameters(self):
        self.hparams = {
            'task' : self.args.task,
            'lowercasing' : self.args.lowercasing,
            'normalize_digits' : self.args.normalize_digits,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_model(self):
        model_path = self.args.model_path
        array = model_path.split('.pkl')
        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.show_hyperparameters()

        # model
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        self.log('Load model: {}\n'.format(model_path))


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

                if (key == 'lowercasing' or
                    key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']


    def setup_data_loader(self):
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        self.data_loader = attribute_annotation_data_loader.AttributeAnnotationDataLoader(
            token_index=self.args.token_index,
            label_index=self.args.label_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
            attr_chunking_flags=common.get_attribute_boolvalues(
                self.args.attr_chunking_flags, len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            attr_delim=self.args.attr_delim,
            lowercasing=self.hparams['lowercasing'],
            normalize_digits=self.hparams['normalize_digits'],
        )


    def setup_classifier(self):
        predictor = models.attribute_annotator.AttributeAnnotator()
        self.classifier = classifiers.pattern_matcher.PatternMatcher(predictor)


    def setup_optimizer(self):
        pass


    def setup_evaluator(self, evaluator=None):
        ignored_labels = set()
        for label in self.args.ignored_labels.split(','):
            label_id = self.dic.tables[constants.ATTR_LABEL(0)].get_id(label)
            if label_id >= 0:
                self.args.ignored_labels.add(label_id)
                
        self.log('Setup evaluator: labels to be ignored={}\n'.format(ignored_labels))
        self.evaluator = AccuracyEvaluator(
            ignore_head=False, ignored_labels=ignored_labels)


    def gen_inputs(self, data, evaluate=True):
        xs = data.inputs[0]
        ps = data.inputs[0] if data.inputs[0] else None
        ls = data.outputs[0] if evaluate else None
        if evaluate:
            return xs, ps, ls
        else:
            return xs, ps


    def decode(self, rdata, file=sys.stdout):
        org_tokens = rdata.orgdata[0]
        org_attrs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None
        n_ins = len(org_tokens)

        timer = util.Timer()
        timer.start()
        inputs = self.gen_inputs(rdata, evaluate=False)
        self.decode_batch(*inputs, org_tokens=org_tokens, org_attrs=org_attrs, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, *inputs, org_tokens=None, org_attrs=None, file=sys.stdout):
        id2label = self.dic.tables[constants.SEM_LABEL].id2str
        ls = self.classifier.decode(*inputs)

        use_attr = org_attrs is not None
        if not use_attr:
            org_attrs = [None] * len(org_tokens)

        for x_str, p_str, l in zip(org_tokens, org_attrs, ls):
            l_str = [id2label[int(li)] for li in l] 

            if use_attr:
                res = ['{}\t{}\t{}'.format(
                    xi_str, pi_str, li_str) for xi_str, pi_str, li_str in zip(x_str, p_str, l_str)]
            else:
                res = ['{}\t{}'.format(xi_str, li_str) for xi_str, li_str in zip(x_str, l_str)]
            res = '\n'.join(res).rstrip()

            print(res+'\n', file=file)
            # print(res, file=file)


    def run_epoch(self, data, train=True):
        classifier = self.classifier
        evaluator = self.evaluator

        inputs = self.gen_inputs(data)
        xs = inputs[0]
        n_sen = len(xs)

        golds = inputs[self.label_begin_index]
        if train:
            self.classifier.train(*inputs)
        ret = self.classifier.decode(*inputs[:self.label_begin_index])
        counts = self.evaluator.calculate(*[xs], *[golds], *[ret])

        if train:
            self.log('\n<training result>')
            res = evaluator.report_results(n_sen, counts, file=self.logger)
            self.report('train\t%s' % res)

            if self.args.devel_data:
                self.log('\n<development result>')
                v_res = self.run_epoch(self.dev, train=False)
                self.report('devel\t%s' % v_res)

            # save model
            if not self.args.quiet:
                mdl_path = '{}/{}.pkl'.format(constants.MODEL_DIR, self.start_time)
                self.log('Save the model: %s\n' % mdl_path)
                self.report('[INFO] Save the model: %s\n' % mdl_path)
                with open(mdl_path, 'wb') as f:
                    pickle.dump(self.classifier, f)

            if not self.args.quiet:
                self.reporter.close() 
                self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')

        res = None if train else evaluator.report_results(n_sen, counts, file=self.logger)
        return res


    def run_train_mode(self):
        # save model
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.MODEL_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

            dic_path = '{}/{}.s2i'.format(constants.MODEL_DIR, self.start_time)
            with open(dic_path, 'wb') as f:
                pickle.dump(self.dic, f)
            self.log('Save string2index table: {}'.format(dic_path))

        # training
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('Start epoch: {}\n'.format(time))
        self.report('[INFO] Start epoch at {}'.format(time))

        self.run_epoch(self.train, train=True)

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            inputs = self.gen_inputs(rdata, evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)
