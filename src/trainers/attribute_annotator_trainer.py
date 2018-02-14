import sys
import pickle
from datetime import datetime

import util
import constants
import data_io
import classifiers
import models.attribute_annotator
from trainers.trainer import Trainer


class AttributeAnnotatorTrainer(Trainer):
    def __init__(self, args, logger=sys.stderr):
        args.task = constants.TASK_ATTR
        super().__init__(args, logger)
        self.label_begin_index = 2


    def init_hyperparameters(self):
        self.hparams = {
            'task' : self.args.task,
            'subpos_depth' : self.args.subpos_depth,
            'lowercase' : self.args.lowercase,
            'normalize_digits' : self.args.normalize_digits,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.classifier = classifiers.init_classifier(self.task, self.hparams, self.dic)


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

                if (key == 'subpos_depth'):
                    val = int(val)

                if (key == 'lowercase' or
                    key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']


    def load_model(self):
        model_path = self.args.model_path
        array = model_path.split('.')
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
        self.init_model()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.classifier = model
        self.log('Load model parameters: {}\n'.format(model_path))


    def update_model(self):
        # do nothing
        pass


    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold_attr: {} ... {}\n'.format(train.outputs[1][0], train.outputs[1][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        if self.dic.has_table(constants.POS_LABEL):
            id2pos = {v:k for k,v in self.dic.tables[constants.POS_LABEL].str2id.items()}
            self.log('# pos labels: {}\n'.format(id2pos))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def gen_inputs(self, data, evaluate=True):
        ws = data.inputs[0]
        ps = data.outputs[0] if data.outputs[0] else None
        ls = data.outputs[1] if evaluate else None
        return ws, ps, ls


    def setup_evaluator(self):
        tmp = self.args.ignored_labels
        self.args.ignored_labels = set()

        for label in tmp.split(','):
            label_id = self.dic.tables[constants.ATTR_LABEL].get_id(label)
            if label_id >= 0:
                self.args.ignored_labels.add(label_id)
                
        self.log('Setup evaluator: labels to be ignored={}\n'.format(self.args.ignored_labels))
        self.evaluator = classifiers.init_evaluator(self.task, self.dic, self.args.ignored_labels)


    def setup_optimizer(self):
        # do nothing
        pass


    def run_step(self, data, train=True):
        ws, ps, ls = self.gen_inputs(data)
        if train:
            self.classifier.train(ws, ps, ls)
        ys = self.classifier.decode(ws, ps)
        counts = self.evaluator.calculate(*(ws, ls, ys))
        return counts


    def decode(self, rdata, use_pos=True, file=sys.stdout):
        use_pos = True if len(rdata.outputs) > 0 else None
        ws = rdata.inputs[0]
        ps = rdata.outputs[0] if use_pos else None
        orgseqs = rdata.orgdata[0]
        orgposs = rdata.orgdata[1] if use_pos else None
        n_ins = len(ws)

        timer = util.Timer()
        timer.start()
        self.decode_batch(ws, ps, orgseqs, orgposs, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, ws, ps, orgseqs, orgposs, file=sys.stdout):
        ls = self.classifier.decode(ws, ps)

        use_pos = ps is not None
        if not use_pos:
            orgposs = [None] * len(ws)

        id2label = self.dic.tables[constants.ATTR_LABEL].id2str

        for x_str, p_str, l in zip(orgseqs, orgposs, ls):
            l_str = [id2label[int(li)] for li in l] 

            if use_pos:
                res = ['{}\t{}\t{}'.format(
                    xi_str, pi_str, li_str) for xi_str, pi_str, li_str in zip(x_str, p_str, l_str)]
            else:
                res = ['{}\t{}'.format(xi_str, li_str) for xi_str, li_str in zip(x_str, l_str)]
            res = '\n'.join(res).rstrip()
                    
            print(res, file=file)


    def run_train_mode(self):
        self.n_iter = self.args.epoch_begin

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
        self.log('Start epoch 1: {}\n'.format(time))
        self.report('[INFO] Start epoch 1 at {}'.format(time))

        t_counts = self.run_step(self.train)
        self.log('\n### Finish training (%s examples)' % (len(self.train.inputs[0])))
        self.log('<training result>')
        t_res = self.evaluator.report_results(
            len(self.train.inputs[0]), t_counts, loss=None, file=self.logger)
        self.report('train\t%s' % (t_res))

        if self.args.devel_data:
            self.log('\n<development result>')
            d_counts = self.run_step(self.dev, train=False)
            d_res = self.evaluator.report_results(
                len(self.dev.inputs[0]), d_counts, loss=None, file=self.logger)
            self.report('devel\t%s' % (d_res))

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

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_eval_mode(self):
        self.log('<test result>')
        counts = self.run_step(self.test, train=False)
        res = self.evaluator.report_results(
            len(self.test.inputs[0]), counts, loss=None, file=self.logger)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def load_decode_data(self):
        text_path = (self.args.path_prefix if self.args.path_prefix else '') + self.args.decode_data
        use_pos = constants.POS_LABEL in self.dic.tables
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False

        rdata = data_io.load_decode_data(
            text_path, self.dic, self.task, data_format=self.args.input_data_format,
            use_pos=use_pos, use_subtoken=False, subpos_depth=subpos_depth,
            lowercase=lowercase, normalize_digits=normalize_digits)

        self.decode_data = rdata


    def run_interactive_mode(self):
        use_pos = constants.POS_LABEL in self.dic.tables
        subpos_depth = self.hparams['subpos_depth'] if 'subpos_depth' in self.hparams else -1
        lowercase = self.hparams['lowercase'] if 'lowercase' in self.hparams else False
        normalize_digits = self.hparams['normalize_digits'] if 'normalize_digits' else False

        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip()
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            use_pos_now = use_pos and '/' in line
            rdata = data_io.parse_commandline_input(
                line, self.dic, self.task, use_pos=use_pos_now,
                lowercase=lowercase, normalize_digits=normalize_digits,
                use_subtoken=False, subpos_depth=subpos_depth)

            ws = rdata.inputs[0]
            ps = rdata.outputs[0] if use_pos_now else None
            orgseqs = rdata.orgdata[0]
            orgposs = rdata.orgdata[1] if use_pos_now else None

            self.decode_batch(ws, ps, orgseqs, orgposs)
            if not self.args.output_empty_line:
                print(file=sys.stdout)
