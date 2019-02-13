import sys

from evaluators.common import AccuracyEvaluator, DoubleAccuracyEvaluator


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
