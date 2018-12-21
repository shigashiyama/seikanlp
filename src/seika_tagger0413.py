import sys

import common

from seikanlp import SeikaNLP
from arguments import tagger_arguments
from trainers import hybrid_unit_segmenter_trainer0413


class SeikaTagger0(SeikaNLP):
    def __init__(self):
        super().__init__()


    def get_args(self):
        parser = tagger_arguments.TaggerArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        trainer = hybrid_unit_segmenter_trainer0413.HybridUnitSegmenterTrainer0413(args)
        return trainer


if __name__ == '__main__':
    analyzer = SeikaTagger0()
    analyzer.run()
