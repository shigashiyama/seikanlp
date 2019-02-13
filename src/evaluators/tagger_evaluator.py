from collections import Counter
import sys

import constants
from evaluators.common import Counts


#### vocabulary-based evaluation

class RCountsV(Counts):
    def __init__(self, n_vocabs=1):
        ## index
        # -1       : counts for entity out of all vocabs
        # n (n>=0) : coutns for entity in n-th vocab (but not in (n-1)-th vocab)
        self.index2ngold = Counter({-1:0, 0:0})
        self.index2ncorrect = Counter({-1:0, 0:0})


    def merge(self, counts):
        if not isinstance(counts, RCountsV):
            print('Invalid count object', file=sys.stderr)
            return

        self.index2ngold += counts.index2ngold
        self.index2ncorrect += counts.index2ncorrect


    def count(self, sen, spans_gold, spans_pred, vocabs):
        for span in spans_gold:
            idx = vocabs.get_index(sen, span)
            self.index2ngold[idx] += 1

        for span in spans_pred:
            if span in spans_gold:
                idx = vocabs.get_index(sen, span)
                self.index2ncorrect[idx] += 1


    def get_count_distribution(self):
        ngolds = {}
        ncorrects = {}

        last_index = max(self.index2ngold.keys())
        last_out_key = '{}:out'.format(last_index)
        ngolds[last_out_key] = self.index2ngold[-1]
        ncorrects[last_out_key] = self.index2ncorrect[-1]
        
        for i in range(last_index, -1, -1):
            i_in_key = '{}:in'.format(i)
            ngolds[i_in_key] = self.index2ngold[i]
            ncorrects[i_in_key] = self.index2ncorrect[i]

            i_out_key = '{}:out'.format(i) # already registered
            i_1_out_key = '{}:out'.format(i-1) if i > 0 else 'total'
            ngolds[i_1_out_key] = ngolds[i_in_key] + ngolds[i_out_key] 
            ncorrects[i_1_out_key] = ncorrects[i_in_key] + ncorrects[i_out_key] 

        return ngolds, ncorrects


class FMeasureCalculatorForEachVocab(object):
    def __init__(self, id2label, vocabs):
        self.id2label = id2label
        self.vocabs = vocabs


    def __call__(self, xs, ts, ys):
        counts = RCountsV()
        for x, t, y in zip(xs, ts, ys):
            x_ = [int(xi) for xi in x]
            t_spans = BIES2spans([self.id2label[int(ti)] for ti in t])
            y_spans = BIES2spans([self.id2label[int(yi)] for yi in y])
            counts.count(x_, t_spans, y_spans, self.vocabs)

        return counts


class FMeasureEvaluatorForEachVocab(object):
    def __init__(self, id2label, vocabs):
        self.calculator = FMeasureCalculatorForEachVocab(id2label, vocabs)


    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts
        

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ngolds, ncorrects = counts.get_count_distribution()
        total_ngold = ngolds['total']
        ave_loss = loss / total_ngold
        print('ave loss: %.5f'% ave_loss, file=file)

        if True:
            key = 'total'
            ngold = ngolds[key]
            ncorrect = ncorrects[key]
            r = 1.0 * ncorrect / ngold
            print('Total                  - N_gold, TP, R: %d, %d,\t%.2f' % (
                ngold, ncorrect, 100*r), file=file)

        last_index = len(self.calculator.vocabs)
        for i in range(last_index):
            for key in ('{}:in'.format(i), '{}:out'.format(i)):
                ngold = ngolds[key]
                ncorrect = ncorrects[key]
                r = 1.0 * ncorrect / ngold
                print('%s vocab-%d (%.2f%%) - N_gold, TP, R: %d, %d,\t%.2f' % (
                    'In    ' if 'in' in key else 'Out-of',
                    i, (100.0*ngold/total_ngold), ngold, ncorrect, 100*r), file=file)

        res = '%.2f\t%.4f' % (r, ave_loss)
        return res


#### word length-based evaluation

class FCountsL(Counts):         # count for each length
    def __init__(self, length_limit=-1):
        self.length_limit = length_limit
        self.len2ngold = Counter()
        self.len2npred = Counter()
        self.len2ncorrect = Counter()


    def merge(self, counts):
        if not isinstance(counts, FCountsL):
            print('Invalid count object', file=sys.stderr)
            return

        self.length_limit = max(self.length_limit, counts.length_limit)
        self.len2npred += counts.len2npred
        self.len2ngold += counts.len2ngold
        self.len2ncorrect += counts.len2ncorrect


    def count(self, spans_gold, spans_pred):
        for s in spans_gold:
            l = s[1] - s[0]
            if -1 < self.length_limit < l:
                l = self.length_limit
            self.len2ngold[l] += 1

        for s in spans_pred:
            l = s[1] - s[0]
            if -1 < self.length_limit < l:
                l = self.length_limit
            self.len2npred[l] += 1

            if s in spans_gold:
                self.len2ncorrect[l] += 1


    def get_total_counts(self):
        ngold = sum(self.len2ngold.values())
        npred = sum(self.len2npred.values())
        ncorrect = sum(self.len2ncorrect.values())
        return ngold, npred, ncorrect


class FMeasureCalculatorForEachLength(object):
    def __init__(self, id2label):
        self.id2label = id2label


    def __call__(self, xs, ts, ys):
        counts = FCountsL(length_limit=constants.DEFAULT_LENGTH_LIMIT)
        for _, t, y in zip(xs, ts, ys):
            t_spans = BIES2spans([self.id2label[int(ti)] for ti in t])
            y_spans = BIES2spans([self.id2label[int(yi)] for yi in y])
            counts.count(t_spans, y_spans)

        return counts


class FMeasureEvaluatorForEachLength(object):
    def __init__(self, id2label):
        self.calculator = FMeasureCalculatorForEachLength(id2label)


    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts
        

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        total_ngold, total_npred, total_ncorrect = counts.get_total_counts()
        ave_loss = loss / total_ngold
        print('ave loss: %.5f'% ave_loss, file=file)

        max_l = max(set(counts.len2ngold.keys()) | set(counts.len2npred.keys()))
        for l in range(max_l+1):
            if not l in counts.len2ngold and not l in counts.len2npred:
                continue

            ngold = counts.len2ngold[l]
            npred = counts.len2npred[l]
            ncorrect = counts.len2ncorrect[l]
            p, r, f = calculate_PRF(ngold, npred, ncorrect)
            print('Len=%2d (%.2f%%) - N_gold, N_pred, TP, P, R, F: %d, %d, %d,\t%.2f, %.2f, %.2f' % (
                l, (100.0*ngold/total_ngold), ngold, npred, ncorrect, 100*p, 100*r, 100*f), file=file)

        if True:
            p, r, f = calculate_PRF(total_ngold, total_npred, total_ncorrect)
            print('Total          - N_gold, N_pred, TP, P, R, F: %d, %d, %d,\t%.2f, %.2f, %.2f' % (
                total_ngold, total_npred, total_ncorrect, 100*p, 100*r, 100*f), file=file)

        res = '%.2f\t%.2f\t%.2f\t%.4f' % (p, r, f, ave_loss)
        return res


def BIES2spans(seq):
    spans = []
    begin = 0
    for i, v in enumerate(seq):
        if v == 'B':
            begin = i
        elif v == 'E':
            end = i+1
            spans.append((begin, end))
            begin = end
        elif v == 'S':
            begin = i
            end = i+1
            spans.append((begin, end))
            begin = end
    return spans


def calculate_PRF(n_gold, n_pred, n_cor):
    p = 1.0 * n_cor / n_pred
    r = 1.0 * n_cor / n_gold
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f
