import os
import sys
import re
import argparse
import subprocess

import conlleval
sys.path.append('src')
from data import data_loader
import evaluators


class Evaluater(object):
    def __init__(self, pred_path=None):
        self.fp = open(pred_path)


    def close(self):
        if self.fp:
            self.fp.close()


    def get_predicted_result(self, *args):
        res = self.fp.readline()
        res = res.rstrip(' \t')
        res = re.sub(' +', ' ', res)
        return res


    def parse_result(self, res):
        chars = []
        bounds = []
        for word in res.split(' '):
            chars.extend([word[i] for i in range(len(word))])
            bounds.extend([data_loader.get_label_BI(i, None) for i in range(len(word))])
        return chars, bounds


    def iterate_sentences(self, gold_path):
        print('read file:', gold_path)
        with open(gold_path) as fg:
            g_chars = []
            g_bounds = []
        
            for gline in fg:
                # 文頭末尾の空白改行コードを除去、重複する空白を除去
                gline = gline.rstrip(' \t')
                gline = re.sub(' +', ' ', gline)
                if not gline:
                    continue

                g_chars, g_bounds = self.parse_result(gline)
                pres = self.get_predicted_result(gline)
                _, p_bounds = self.parse_result(pres)
                # print('g', gline)
                # print('g', g_chars)
                # print('g', g_bounds)
                # print('p', pres)
                # print('p', _)
                # print('p', p_bounds)

                yield g_chars, g_bounds, p_bounds

        self.close()

        raise StopIteration


    def iterate_tokens_for_eval(self, x, t, y):
        i = 0
        while True:
            if len(t) != len(y):
                print(len(x), x)
                print(len(t), t)
                print(len(y), y)

            if i == len(x):
                raise StopIteration
            x_str = x[i]
            t_str = t[i]
            y_str = y[i]        
            yield [x_str, t_str, y_str]
        
            i += 1


    def run(self, gold_path):
        sen_count = 0
        eval_counts = None

        for g_chars, g_bounds, k_bounds in self.iterate_sentences(gold_path):
            token_iter = self.iterate_tokens_for_eval(g_chars, g_bounds, k_bounds)
            tmp = conlleval.evaluate(token_iter)
            eval_counts = evaluators.merge_counts(eval_counts, tmp)
            sen_count += 1

            # if sen_count % 50000 == 0:
            #     show_results(sen_count, eval_counts)

        # evaluate
        sen_count += 1
        show_results(sen_count, eval_counts)



class KyteaEvaluater(Evaluater):
    def __init__(self, pred_path=None, model_path=None, use_pos=False):
        self.fp = open(pred_path) if pred_path else None
        self.model_path = model_path
        self.use_pos = use_pos


    def get_predicted_result(self, gline=None):
        if self.fp:
            res = self.fp.readline().rstrip(' \t')
        else:
            # run kytea
            sen = ''
            tokens = gline.split(' ')
            for token in tokens:
                sen += token.split('/')[0]

            cmd = 'echo ' + sen + ' | kytea'
            if model_path:
                cmd += ' -model ' + model_path
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            res = out.decode('utf-8').rstrip(' \t')
        
        return res


    def parse_result(self, res):
        chars = []
        bounds = []
        for token in res.split(' '):
            feats = token.split('/')
            word = feats[0]
            pos = feats[1] if self.use_pos else None
            chars.extend([word[i] for i in range(len(word))])
            bounds.extend([data_loader.get_label_BI(i, pos) for i in range(len(word))])
        
        return chars, bounds


def show_results(sen_count, eval_count):
    c = eval_count
    met = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)

    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (sen_count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (met.tp, met.fp, met.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*met.acc, 100.*met.prec, 100.*met.rec, 100.*met.fscore))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmenter', '-s', default='')
    parser.add_argument('--model_path', '-m', default='')
    parser.add_argument('--gold_path', '-g', default='')
    parser.add_argument('--pred_path', '-p', default='')
    parser.add_argument('--use_pos', default=False, action='store_true') 
    args = parser.parse_args()
    print(args)

    if args.segmenter == 'kytea':
        evaluater = KyteaEvaluater(args.pred_path, args.model_path, args.use_pos)
    else:
        evaluater = Evaluater(args.pred_path)

    print('<result>')
    evaluater.run(args.gold_path)

