import os
import sys
sys.path.append(os.pardir())

import argparse
import subprocess

import data
from eval.conlleval import conlleval


class Evaluater(object):
    def __init__():
        pass

    def get_predicted_result():
        return None

    def parse_result():
        return None

    def close():
        pass

    def iterate_sentences(gold_path):
        print('read file:', gold_path)
        with open(gold_path) as fg:
            g_chars = []
            g_bounds = []
        
            for gline in fg:
                gline = gline.rstrip()
                g_chars, g_bounds = parse_result(gline)
                pres = self.get_predicted_result(gline)
                _, p_bounds = self.parse_result(res)

                yield g_chars, g_bounds, k_bounds

        self.close()

        raise StopIteration


    def iterate_tokens_for_eval(x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = x[i]
            t_str = t[i]
            y_str = y[i]        
            #yield [x_str, t_str, y_str]
            yield x_str, t_str, y_str
        
            i += 1


    def run(gold_path):
        sen_count = 0
        eval_counts = None

        for g_chars, g_bounds, k_bounds in self.iterate_sentences(gold_path):
            token_iter = iterate_tokens_for_eval(g_chars, g_bounds, k_bounds)
            eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(token_iter))
            sen_count += 1

            if sen_count % 50000 == 0:
                show_results(sen_count, eval_counts)

        # evaluate
        sen_count += 1
        show_results(sen_count, eval_counts)



class KyteaEvaluater(Evaluater):
    def __init__(pred_path=None, model_path=None):
        self.fp = open(pred_path) if pred_path else None
        self.model_path = model_path


    def get_predicted_result(gline=None):
        if self.fp:
            res = self.fp.readline().rstrip()
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
            res = out.decode('utf-8').rstrip()
        
        return res


    def parse_result(res):
        chars = []
        bounds = []
        for token in res.split(' '):
            feats = token.split('/')
            word = feats[0]
            # pos = feats[1]
            # read = feats[2]

            chars.extend([word[i] for i in range(len(word))])
            bounds.extend([data.get_label_BI(i, None) for i in range(len(word))])
        
        return chars, bounds


    def close():
        if self.fp:
            fp.close()


class StanfordSegmenterEvaluater(Evaluater):
    def __init__(pred_path=None):
        self.fp = open(pred_path)
    


def show_results(sen_count, eval_count):
    c = eval_count
    acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
    overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)

    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (sen_count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmenter', '-s', default='')
    parser.add_argument('--model_path', '-m', default='')
    parser.add_argument('--gold_path', '-g', default='')
    parser.add_argument('--pred_path', '-p', default='') 
    args = parser.parse_args()
    print(args, '\n')

    if args.segmenter == 'kytea':
        evaluater = KyteaEvaluater(args.pred_path, args.model_path)
    elif args.segmenter == 'stanford':
        evaluater = StanfordSegmenter(args.pred_path)

    print('<result>')
    evaluater.run(args.gold_path)
