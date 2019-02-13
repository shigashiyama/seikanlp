import sys

import common

from seikanlp import SeikaNLP
from arguments import attribute_annotator_arguments
from trainers import attribute_annotator_trainer


class SeikaAttributeAnnotator(SeikaNLP):
    def __init__(self):
        super().__init__()


    def get_args(self):
        parser = attribute_annotator_arguments.AttributeAnnotatorArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        trainer = attribute_annotator_trainer.AttributeAnnotatorTrainer(args)
        return trainer


if __name__ == '__main__':
    analyzer = SeikaAttributeAnnotator()
    analyzer.run()
