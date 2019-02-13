import chainer

import constants
import models.util


class Classifier(chainer.link.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)
    

    def load_pretrained_embedding_layer(self, dic, external_model, finetuning=False):
        id2unigram = dic.tables[constants.UNIGRAM].id2str
        usage = self.predictor.pretrained_embed_usage

        if usage == models.util.ModelUsage.INIT:
            models.util.load_pretrained_embedding_layer(
                id2unigram, self.predictor.unigram_embed, external_model, finetuning=finetuning)

        elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
            models.util.load_pretrained_embedding_layer(
                id2unigram, self.predictor.pretrained_unigram_embed, external_model, finetuning=finetuning)
