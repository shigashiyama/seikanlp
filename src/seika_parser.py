import sys

import common
import constants
import arguments

from seikanlp import SeikaNLP
from arguments import parser_arguments
from trainers import parser_trainer


class SeikaParser(SeikaNLP):
    def __init__(self):
        super().__init__()


    def get_args(self):
        parser = parser_arguments.ParserArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        trainer = parser_trainer.ParserTrainer(args)
        
        return trainer


if __name__ == '__main__':
    analyzer = SeikaParser()
    analyzer.run()
