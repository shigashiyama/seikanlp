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

    elif isinstance(c1, DFCounts) or isinstance(c2, DFCounts):
        if c1 is None:
            return c2
        if c2 is None:
            return c1

        c = DFCounts()
        c.l1 = merge_counts(c1.l1, c2.l1)
        c.l2 = merge_counts(c1.l2, c2.l2)        

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


class DFCounts(object):
    def __init__(self):
        self.l1 = conlleval.FCounts()
        self.l2 = conlleval.FCounts()


class AccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels


    def __call__(self, ts, ys):
        counts = ACounts()
        for t, y in zip(ts, ys):
            if self.ignore_head:
                t = t[1:]
                y = y[1:]

            for ti, yi in zip(t, y):
                if int(ti) in self.ignored_labels:
                    continue
                
                counts.total += 1 
                if ti == yi:
                    counts.correct += 1 

        return counts


class DoubleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels
        

    def __call__(self, t1s, t2s, y1s, y2s):
        counts = DACounts()
        for t1, t2, y1, y2 in zip(t1s, t2s, y1s, y2s):

            if self.ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                y1 = y1[1:]
                y2 = y2[1:]                

            for t1i, t2i, y1i, y2i in zip(t1, t2, y1, y2):
                if int(t2i) in self.ignored_labels:
                    continue

                counts.l1.total += 1
                counts.l2.total += 1 
                if t1i == y1i:
                    counts.l1.correct += 1

                    if t2i == y2i: # depend on t1i equals to y1i
                        counts.l2.correct += 1

        return counts


class TripleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels
        

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
                if int(t3i) in self.ignored_labels:
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
    def __init__(self, id2label):
        self.id2label = id2label


    def __call__(self, xs, ts, ys):
        counts = conlleval.FCounts()
        for x, t, y in zip(xs, ts, ys):
            generator = self.generate_lines(x, t, y)
            counts = conlleval.evaluate(generator, counts=counts)
        return counts


    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]
            i += 1


class DoubleFMeasureCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label


    def __call__(self, xs, ts, ys):
        counts = DFCounts()

        for x, t, y in zip(xs, ts, ys):
            generator_seg = self.generate_lines_seg(x, t, y)
            generator_tag = self.generate_lines(x, t, y)
            counts.l1 = conlleval.evaluate(generator_seg, counts=counts.l1)
            counts.l2 = conlleval.evaluate(generator_tag, counts=counts.l2)

        return counts


    def generate_lines_seg(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])] 
            tseg_str = t_str.split('-')[0]
            yseg_str = y_str.split('-')[0]

            yield [x_str, tseg_str, yseg_str]
            i += 1


    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                raise StopIteration
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE' 
            y_str = self.id2label[int(y[i])] 

            yield [x_str, t_str, y_str]
            i += 1


class SegmenterEvaluator(object):
    def __init__(self, id2label):
        self.calculator = FMeasureCalculator(id2label)


    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts


    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.token_counter
        met = conlleval.calculate_metrics(
            counts.correct_chunk, counts.found_guessed, counts.found_correct)
        
        print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.token_counter, counts.found_correct, counts.found_guessed), file=file)
        print('TP, FP, FN: %d %d %d' % (met.tp, met.fp, met.fn), file=file)
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
              (100.*met.acc, 100.*met.prec, 100.*met.rec, 100.*met.fscore), file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % (
            (100.*met.acc), (100.*met.prec), (100.*met.rec), (100.*met.fscore), ave_loss)
        return res


class JointSegmenterEvaluator(object):
    def __init__(self, id2label):
        self.calculator = DoubleFMeasureCalculator(id2label)


    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts


    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.token_counter
        met1 = conlleval.calculate_metrics(
            counts.l1.correct_chunk, counts.l1.found_guessed, counts.l1.found_correct)
        met2 = conlleval.calculate_metrics(
            counts.l2.correct_chunk, counts.l2.found_guessed, counts.l2.found_correct)
        
        print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.l1.token_counter, counts.l1.found_correct, counts.l1.found_guessed), 
              file=file)
        print('SEG - TP, FP, FN / A, P, R, F: %d %d %d /\t%6.2f %6.2f %6.2f %6.2f' % (
            met1.tp, met1.fp, met1.fn, 100.*met1.acc, 100.*met1.prec, 100.*met1.rec, 100.*met1.fscore),
              file=file)
        print('TAG - TP, FP, FN / A, P, R, F: %d %d %d /\t%6.2f %6.2f %6.2f %6.2f' % (
            met2.tp, met2.fp, met2.fn, 100.*met2.acc, 100.*met2.prec, 100.*met2.rec, 100.*met2.fscore),
              file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % (
            (100.*met1.acc), (100.*met1.prec), (100.*met1.rec), (100.*met1.fscore), 
            (100.*met2.acc), (100.*met2.prec), (100.*met2.rec), (100.*met2.fscore), ave_loss)
        return res


class AccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = AccuracyCalculator(ignore_head, ignored_labels)


    def calculate(self, *inputs):
        ts = inputs[1]
        ys = inputs[2]
        counts = self.calculator(ts, ys)
        return counts


    def report_results(self, sen_counter, counts, loss=None, file=sys.stderr):
        ave_loss = loss / counts.total if loss is not None else None
        acc = 1.*counts.correct / counts.total
        
        if ave_loss is not None:
            print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, correct: {} {} {}'.format(
            sen_counter, counts.total, counts.correct), file=file)
        print('A:%6.2f' % (100.*acc), file=file)

        if ave_loss is not None:
            res = '%.2f\t%.4f' % ((100.*acc), ave_loss)
        else:
            res = '%.2f' % (100.*acc)

        return res


class DoubleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = DoubleAccuracyCalculator(ignore_head, ignored_labels)


    def calculate(self, *inputs):
        t1s = inputs[1]
        t2s = inputs[2]
        y1s = inputs[3]
        y2s = inputs[4]
        counts = self.calculator(t1s, t2s, y1s, y2s)
        return counts


class TripleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = TripleAccuracyCalculator(ignore_head, ignored_labels)


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
    def __init__(self, ignore_head=False, ignored_labels=set()):
        super().__init__(ignore_head=ignore_head, ignored_labels=ignored_labels)


class ParserEvaluator(AccuracyEvaluator):
    def __init__(self, ignore_head=True):
        super().__init__(ignore_head=ignore_head, ignored_labels=set())


class TypedParserEvaluator(DoubleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignored_labels=set()):
        super().__init__(ignore_head=ignore_head, ignored_labels=ignored_labels)


    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.total
        uas = 1.*counts.l1.correct / counts.l1.total
        las = 1.*counts.l2.correct / counts.l2.total
        
        print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, correct head, correct label: {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct), file=file)
        print('UAS, LAS:%6.2f %6.2f' % ((100.*uas), (100.*las)), file=file)

        res = '%.2f\t%.2f\t%.4f' % ((100.*uas), (100.*las), ave_loss)
        return res


class TaggerParserEvaluator(DoubleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignored_labels=set()):
        super().__init__(ignore_head=ignore_head, ignored_labels=ignored_labels)


    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.total
        uas = 1.*counts.l1.correct / counts.l1.total
        las = 1.*counts.l2.correct / counts.l2.total
        
        print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, correct head, correct label: {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct), file=file)
        print('POS, UAS:%6.2f %6.2f' % ((100.*uas), (100.*las)), file=file)

        res = '%.2f\t%.2f\t%.4f' % ((100.*uas), (100.*las), ave_loss)
        return res


class TaggerTypedParserEvaluator(TripleAccuracyEvaluator):
    def __init__(self, ignore_head=True, ignored_labels=set()):
        super().__init__(ignore_head=ignore_head, ignored_labels=ignored_labels)


    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.total
        pos = 1.*counts.l1.correct / counts.l1.total
        uas = 1.*counts.l2.correct / counts.l2.total
        las = 1.*counts.l3.correct / counts.l3.total
        
        print('ave loss: %.5f'% ave_loss, file=file)
        print('sen, token, correct pos, correct head, correct label: {} {} {} {} {}'.format(
            sen_counter, counts.l1.total, counts.l1.correct, counts.l2.correct, counts.l3.correct), 
              file=file)
        print('POS, UAS, LAS:%6.2f %6.2f %6.2f' % ((100.*pos), (100.*uas), (100.*las)), file=file)

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
