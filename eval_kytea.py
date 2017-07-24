import argparse
import subprocess
from subprocess import PIPE

import util
import sys
from conlleval import conlleval


def train_kytea(train_path, model_path, dict_path):
    cmd = 'train-kytea -full ' + train_path + ' -model ' + model_path
    if dict_path:
        cmd += ' -dict ' + dict_path
    p = subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    print(out.decode('utf-8').rstrip())
    return


def eval_kytea(gold_path, pred_path='', model_path=''):
    sen_count = 0
    eval_counts = None

    for g_chars, g_bounds, k_bounds in sentence_iterator(
            gold_path, pred_path=pred_path, model_path=model_path):

        iter = generate_lines(g_chars, g_bounds, k_bounds)
        eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(iter))
        sen_count += 1

        if sen_count % 50000 == 0:
            c = eval_counts
            acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
            overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
            print('end: %d sentences' % sen_count)
            print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
                  (sen_count, c.token_counter, c.found_correct, c.found_guessed))
            print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
            print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
                  (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))
            print()

            # log
            # sen = ''
            # fo.write('sen %d: %s\n' % (sen_count, sen))
            # fo.write('out: %s\n' % k_res)
            # iter = generate_lines(g_chars, g_bounds, k_bounds)
            # for token in iter:
            #     fo.write('%s\t%s\t%s\n' % (token[0], token[1], token[2]))
            # fo.write('\n')

    # evaluate
    sen_count += 1
    c = eval_counts
    acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
    overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (sen_count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))


def parse_result(res):
    chars = []
    bounds = []
    for token in res.split(' '):
        feats = token.split('/')
        word = feats[0]
        # pos = feats[1]
        # read = feats[2]

        chars.extend([word[i] for i in range(len(word))])
        bounds.extend([util.get_label_BI(i, None) for i in range(len(word))])
        
    return chars, bounds


def generate_lines(x, t, y):
    i = 0
    while True:
        if i == len(x):
            raise StopIteration
        x_str = x[i]
        t_str = t[i]
        y_str = y[i]        
        yield [x_str, t_str, y_str]
        
        i += 1


def sentence_iterator(gold_path, pred_path='', model_path=''):
    fp = open(pred_path) if pred_path else None

    print('read file:', gold_path)
    with open(gold_path) as fg:
        g_chars = []
        g_bounds = []
        
        for line in fg:
            line = line.rstrip()
            g_chars, g_bounds = parse_result(line)

            if fp:
                res = fp.readline().rstrip()
            else:
                # run kytea
                sen = ''
                tokens = line.split(' ')
                for token in tokens:
                    sen += token.split('/')[0]

                cmd = 'echo ' + sen + ' | kytea'
                if model_path:
                    cmd += ' -model ' + model_path
                p = subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
                out, err = p.communicate()
                res = out.decode('utf-8').rstrip()
            _, k_bounds = parse_result(res)

            yield [g_chars, g_bounds, k_bounds]

    if fp:
        fp.close()

    raise StopIteration
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', default='')
    parser.add_argument('--dict', default='')
    parser.add_argument('--org_data_dir', default='')
    parser.add_argument('--train', type=int, default=-1)
    parser.add_argument('--eval', type=int, default=-1)
    parser.add_argument('--data_dir', '-d', default='')
    parser.add_argument('--fileid', '-f', default='')
    parser.add_argument('--pred_dir', '-p', default='') 
    parser.add_argument('--out_dir', '-o', default='')
    args = parser.parse_args()
    print(args, '\n')

    # preprocess data
    exts = ['_train.tsv', '_val.tsv', '_test.tsv']

    if args.preprocess:
        for ext in exts:
            input = args.org_data_dir + args.fileid + ext
            output = args.data_dir + args.fileid + ext
            util.process_bccwj_data_for_kytea(input, output)

    train_gold = args.data_dir + args.fileid + '_train.tsv'
    val_gold = args.data_dir + args.fileid + '_val.tsv'
    test_gold = args.data_dir + args.fileid + '_test.tsv'
    model_path = args.out_dir + args.fileid + '.mod'
    debug_path = args.out_dir + args.fileid + '.debug'

    if args.train > 0:
        train_kytea(train_gold, model_path, args.dict)

    if args.eval > 0:
        train_pred = args.pred_dir + args.fileid + '_train.pred' if args.pred_dir else ''
        val_pred = args.pred_dir + args.fileid + '_val.pred' if args.pred_dir else ''
        test_pred = args.pred_dir + args.fileid + '_test.pred' if args.pred_dir else ''

        # print('<training result>')
        # eval_kytea(gold_path=train_gold, pred_path=train_pred, model_path=model_path)
        # print()
        print('<validation result>')
        eval_kytea(gold_path=val_gold, pred_path=val_pred, model_path=model_path)

        print('<test result>')
        eval_kytea(gold_path=test_gold, pred_path=test_pred, model_path=model_path)
