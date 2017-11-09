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
        print('Invalid count object', sys.stderr)
        return None


class ACounts(object):
    def __init__(self):
        self.total = 0
        self.correct = 0


class DACounts(object):
    def __init__(self):
        self.l1 = ACounts()
        self.l2 = ACounts()


class AccuracyCalculator(object):
    def __call__(self, ts, ys, ignore_head=False):
        counts = ACounts()
        for t, y in zip(ts, ys):
            if ignore_head:
                t = t[1:]
                y = y[1:]

            for ti, yi in zip(t, y):
                counts.total += 1 
                if ti == yi:
                    counts.correct += 1 

        return counts


class DoubleAccuracyCalculator(object):
    def __call__(self, t1s, t2s, y1s, y2s, ignore_head=False):
        counts = DACounts()
        for t1, t2, y1, y2 in zip(t1s, t2s, y1s, y2s):
            # print(t1)
            # print(y1)
            # print(t2)
            # print(y2)

            if ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                y1 = y1[1:]
                y2 = y2[1:]                

            for t1i, t2i, y1i, y2i in zip(t1, t2, y1, y2):
                counts.l1.total += 1
                counts.l2.total += 1 
                if t1i == y1i:
                    counts.l1.correct += 1

                    if t2i == y2i:
                        counts.l2.correct += 1

        return counts


class FMeasureCalculator(object):
    def __init__(self, label_indices):
        self.label_indices = label_indices


    def __call__(self, xs, ts, ys):
        counts = None
        for x, t, y in zip(xs, ts, ys):
            generator = self.generate_lines(x, t, y)
            eval_counts = conlleval.merge_counts(counts, conlleval.evaluate(generator))
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
    def __init__(self, ignore_head=False):
        self.calculator = AccuracyCalculator()
        self.ignore_head = ignore_head


    def calculate(self, *inputs):
        ts = inputs[1]
        ys = inputs[2]
        counts = self.calculator(ts, ys, ignore_head=self.ignore_head)
        return counts


    def report_results(self, sen_counter, counts, loss, stream=sys.stderr):
        ave_loss = loss / counts.total
        acc = 1.*counts.correct / counts.total
        
        print('ave loss: %.5f'% ave_loss, file=stream)
        print('sen, token, correct: {} {} {}'.format(
            sen_counter, counts.total, counts.correct), file=stream)
        print('A:%6.2f' % (100.*met.acc), file=stream)

        res = '%.2f\t%.4f' % ((100.*met.acc), ave_loss)
        return res


class TaggerEvaluator(AccuracyEvaluator):
    def __init__(self):
        super(TaggerEvaluator, self).__init__(ignore_head=False)


class ParserEvaluator(AccuracyEvaluator):
    def __init__(self):
        super(ParserEvaluator, self).__init__(ignore_head=True)


class TypedParserEvaluator(object):
    def __init__(self):
        self.calculator = DoubleAccuracyCalculator()


    def calculate(self, *inputs):
        ths = inputs[1]
        tls = inputs[2]
        yhs = inputs[3]
        yls = inputs[4]
        counts = self.calculator(ths, tls, yhs, yls, ignore_head=True)
        return counts


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
