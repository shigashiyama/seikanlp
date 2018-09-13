import sys

import common
import constants
import arguments

from seikanlp import SeikaNLP
from arguments import tagger_arguments
from trainers import tagger_trainer, hybrid_unit_segmenter_trainer


class SeikaTagger(SeikaNLP):
    def __init__(self):
        super().__init__()


    def get_args(self):
        parser = tagger_arguments.TaggerArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        if args.tagging_unit == 'single':
            trainer = tagger_trainer.TaggerTrainer(args)
        else:
            trainer = hybrid_unit_segmenter_trainer.HybridUnitSegmenterTrainer(args)
        
        return trainer


if __name__ == '__main__':
    analyzer = SeikaTagger()
    analyzer.run()
