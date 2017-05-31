import subprocess
from subprocess import PIPE

import util
from conlleval import conlleval


def run_and_eval_kytea(in_path, out_path):
    print("Read file:", in_path)

    sen_count = 0
    sen = ''
    with open(in_path) as fi, open(out_path, 'w') as fo:
        bof = True
        g_chars = []
        g_bounds = []
        eval_counts = None
        
        for line in fi:
            line = line.rstrip('\n').split('\t')
            bos = line[0]
            word = line[1]
            
            if not bof and bos == 'B':
                sen_count += 1

                # run kytea on the previous sentence and evaluate it
                k_res, k_bounds = run_kytea_for_word_segmentation(sen)
                iter = generate_lines(g_chars, g_bounds, k_bounds)
                eval_counts = conlleval.merge_counts(eval_counts, conlleval.evaluate(iter))

                if sen_count % 20 == 0:
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

                fo.write('sen %d: %s\n' % (sen_count, sen))
                fo.write('out: %s\n' % k_res)
                iter = generate_lines(g_chars, g_bounds, k_bounds)
                for token in iter:
                    fo.write('%s\t%s\t%s\n' % (token[0], token[1], token[2]))
                fo.write('\n')

                # clear
                sen = ''
                g_chars = []
                g_bounds = []

            sen += word
            g_chars.extend([word[i] for i in range(len(word))])
            g_bounds.extend([util.get_label(i, None) for i in range(len(word))])

            bof = False

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



def run_kytea_for_word_segmentation(sen):
    cmd = 'echo ' + sen + ' | kytea'
    p = subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    res = out.decode('utf-8').rstrip()
    
    chars = []
    bounds = []
    for token in res.split(' '):
        feats = token.split('/')
        word = feats[0]
        # pos = feats[1]
        # read = feats[2]

        # chars.extend([word[i] for i in range(len(word))])
        bounds.extend([util.get_label(i, None) for i in range(len(word))])
        
    # return chars, bounds
    return res, bounds

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

            
if __name__=='__main__':
    dpath='/home/shigashi/data_shigashi/work/work_chainer/wordseg/bccwj_data/Disk4/processed/ma/'
    sample_path = dpath+'LBb_train_small.tsv'
    train_path = dpath+'LBb_train.tsv'
    val_path = dpath+'LBb_val.tsv'
    test_path = dpath+'LBb_test.tsv'

    debug_path = '/home/shigashi/data_shigashi/work/work_chainer/wordseg/out_kytea/res_sample.txt'

    run_and_eval_kytea(sample_path, debug_path)
