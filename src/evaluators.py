import sys

from tools import conlleval


def merge_counts(c1, c2):
    if isinstance(c1, ACounts) or isinstance(c2, ACounts):
        if c1 is None:
            return c2
        if c2 is None:
            return c1

        c = ACounts()
        c.total = c1.total + c2.total
        c.correct = c1.correct + c2.correct
        return c

    elif isinstance(c1, DACounts) or isinstance(c2, DACounts):
        if c1 is None:
            return c2
        if c2 is None:
            return c1
            
        c = DACounts()
        c.l1 = merge_counts(c1.l1, c2.l1)
        c.l2 = merge_counts(c1.l2, c2.l2)        
        return c


    elif isinstance(c1, TACounts) or isinstance(c2, TACounts):
        if c1 is None:
            return c2
        if c2 is None:
            return c1
            
        c = TACounts()
        c.l1 = merge_counts(c1.l1, c2.l1)
        c.l2 = merge_counts(c1.l2, c2.l2)        
        c.l3 = merge_counts(c1.l3, c2.l3)
        return c


    elif isinstance(c1, conlleval.FCounts) or isinstance(c2, conlleval.FCounts):
        if c1 is None:
            return c2
        if c2 is None:
            return c1

        c = conlleval.FCounts()
        c.correct_chunk = c1.correct_chunk + c2.correct_chunk
        c.correct_tags = c1.correct_tags + c2.correct_tags
        c.found_correct = c1.found_correct + c2.found_correct
        c.found_guessed = c1.found_guessed + c2.found_guessed
        c.token_counter = c1.token_counter + c2.token_counter

        #TODO merge counts for each type

        return c

    else:
        print('Invalid count object', file=sys.stderr)
        return None


class ACounts(object):
    def __init__(self):
        self.total = 0
        self.correct = 0


class DACounts(object):
    def __init__(self):
        self.l1 = ACounts()
        self.l2 = ACounts()


class TACounts(object):
    def __init__(self):
        self.l1 = ACounts()
        self.l2 = ACounts()
        self.l3 = ACounts()


class AccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.ignore_head = ignore_head
        self.ignore_labels = ignore_labels


    def __call__(self, ts, ys):
        counts = ACounts()
        for t, y in zip(ts, ys):
            if self.ignore_head:
                t = t[1:]
                y = y[1:]

            for ti, yi in zip(t, y):
                if int(ti) in self.ignore_labels:
                    continue
                
                counts.total += 1 
                if ti == yi:
                    counts.correct += 1 

        return counts


class DoubleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.ignore_head = ignore_head
        self.ignore_labels = ignore_labels
        

    def __call__(self, t1s, t2s, y1s, y2s):
        counts = DACounts()
        for t1, t2, y1, y2 in zip(t1s, t2s, y1s, y2s):

            if self.ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                y1 = y1[1:]
                y2 = y2[1:]                

            for t1i, t2i, y1i, y2i in zip(t1, t2, y1, y2):
                if int(t2i) in self.ignore_labels:
                    continue

                counts.l1.total += 1
                counts.l2.total += 1 
                if t1i == y1i:
                    counts.l1.correct += 1

                    if t2i == y2i: # depend on t1i equals to y1i
                        counts.l2.correct += 1

        return counts


class TripleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.ignore_head = ignore_head
        self.ignore_labels = ignore_labels
        

    def __call__(self, t1s, t2s, t3s, y1s, y2s, y3s):
        counts = TACounts()
        for t1, t2, t3, y1, y2, y3 in zip(t1s, t2s, t3s, y1s, y2s, y3s):

            if self.ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                t3 = t3[1:]
                y1 = y1[1:]
                y2 = y2[1:]
                y3 = y3[1:]

            for t1i, t2i, t3i, y1i, y2i, y3i in zip(t1, t2, t3, y1, y2, y3):
                if int(t3i) in self.ignore_labels:
                    continue

                counts.l1.total += 1
                counts.l2.total += 1
                counts.l3.total += 1
                if t1i == y1i:
                    counts.l1.correct += 1

                if t2i == y2i:
                    counts.l2.correct += 1

                    if t3i == y3i: # depend on t2i equals to y2i
                        counts.l3.correct += 1

        return counts


class FMeasureCalculator(object):
    def __init__(self, label_indices):
        self.label_indices = label_indices


    def __call__(self, xs, ts, ys):
        counts = None
        for x, t, y in zip(xs, ts, ys):
            generator = self.generate_lines(x, t, y)
            eval_counts = merge_counts(counts, conlleval.evaluate(generator))
        return counts


    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.label_indices.get_label(int(t[i]))
            y_str = self.label_indices.get_label(int(y[i]))

            yield [x_str, t_str, y_str]
            i += 1


class SegmenterEvaluator(object):
    def __init__(self, label_indices):
        self.calculator = FMeasureCalculator()


    def calculate(self, *inputs):
        coutns = self.calculator(*inputs)
        return counts


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.token_counter
        met = conlleval.calculate_metrics(
            counts.correct_chunk, counts.found_guessed, counts.found_correct)
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.token_counter, counts.found_correct, counts.found_guessed), file=stream)
        print('TP, FP, FN: %d %d %d' % (met.tp, met.fp, met.fn), file=stream)
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*met.acc, 100.*met.prec, 100.*met.rec, 100.*met.fscore), file=stream)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % (
            (100.*met.acc), (100.*met.prec), (100.*met.rec), (100.*met.fscore), ave_loss)
        return res


class JointSegmenterEvaluator(object):
    def __init__(self):
        self.calculator = FMeasureCalculator()


    def calculate(self, *inputs):
        # to be implemented
        pass


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        # to be implemented
        pass


class AccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.calculator = AccuracyCalculator(ignore_head, ignore_labels)


    def calculate(self, *inputs):
        ts = inputs[1]
        ys = inputs[2]
        counts = self.calculator(ts, ys)
        return counts


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.total
        acc = 1.*counts.correct / counts.total
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, correct: {} {} {}'.format(
            sen_counter, counts.total, counts.correct), file=stream)
        print('A:%6.2f' % (100.*acc), file=stream)

        res = '%.2f\t%.4f' % ((100.*acc), ave_loss)
        return res


class DoubleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.calculator = DoubleAccuracyCalculator(ignore_head, ignore_labels)


    def calculate(self, *inputs):
        t1s = inputs[1]
        t2s = inputs[2]
        y1s = inputs[3]
        y2s = inputs[4]
        counts = self.calculator(t1s, t2s, y1s, y2s)
        return counts


class TripleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        self.calculator = TripleAccuracyCalculator(ignore_head, ignore_labels)


    def calculate(self, *inputs):
        t1s = inputs[1]
        t2s = inputs[2]
        t3s = inputs[3]
        y1s = inputs[4]
        y2s = inputs[5]
        y3s = inputs[6]
        counts = self.calculator(t1s, t2s, t3s, y1s, y2s, y3s)
        return counts


class TaggerEvaluator(AccuracyEvaluator):
    def __init__(self, ignore_head=False, ignore_labels=set()):
        super(TaggerEvaluator, self).__init__(ignore_head=ignore_head, ignore_labels=ignore_labels)


class ParserEvaluator(AccuracyEvaluator):
    def __init__(self, ignore_head=True, ignore_labels=set()):
        super(ParserEvaluator, self).__init__(ignore_head=ignore_head, ignore_labels=ignore_labels)


class TypedParserEvaluator(DoubleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignore_labels=set()):
        super().__init__(ignore_head=ignore_head, ignore_labels=ignore_labels)


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.l1.total
        uas = 1.*counts.l1.correct / counts.l1.total
        las = 1.*counts.l2.correct / counts.l2.total
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, correct head, correct label: {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct), file=stream)
        print('UAS, LAS:%6.2f %6.2f' % ((100.*uas), (100.*las)), file=stream)

        res = '%.2f\t%.2f\t%.4f' % ((100.*uas), (100.*las), ave_loss)
        return res


class TaggerParserEvaluator(DoubleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignore_labels=set()):
        super().__init__(ignore_head=ignore_head, ignore_labels=ignore_labels)


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.l1.total
        uas = 1.*counts.l1.correct / counts.l1.total
        las = 1.*counts.l2.correct / counts.l2.total
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, correct head, correct label: {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct), file=stream)
        print('POS, UAS:%6.2f %6.2f' % ((100.*uas), (100.*las)), file=stream)

        res = '%.2f\t%.2f\t%.4f' % ((100.*uas), (100.*las), ave_loss)
        return res


class TaggerTypedParserEvaluator(TripleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignore_labels=set()):
        super().__init__(ignore_head=ignore_head, ignore_labels=ignore_labels)


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.l1.total
        pos = 1.*counts.l1.correct / counts.l1.total
        uas = 1.*counts.l2.correct / counts.l2.total
        las = 1.*counts.l3.correct / counts.l3.total
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, correct pos, correct head, correct label: {} {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct, counts.l3.correct), 
              file=stream)
        print('POS, UAS, LAS:%6.2f %6.2f %6.2f' % ((100.*pos), (100.*uas), (100.*las)), file=stream)

        res = '%.2f\t%.2f\t%.4f' % ((100.*uas), (100.*las), ave_loss)
        return res


def ret1():
    return [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

def ret2():
    return [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, -1], [0, 0, -1], [0, 0, -1]]
    ]


if __name__ == '__main__':
    e1 = ParserEvaluator()
    inputs1 = ret1()
    inputs2 = ret2()[1:]
    inputs3 = ret2()[1:]
    e1.calculate(*[inputs1], *inputs2, *inputs3)

    e2 = TypedParserEvaluator()
    inputs1 = ret1()
    inputs2 = ret2()
    inputs3 = ret2()
    e2.calculate(*[inputs1], *inputs2, *inputs3)
