class PatternMatcher(object):
    def __init__(self, predictor):
        self.predictor = predictor


    def train(self, ws, ps, ls):
        if not ps:
            ps = [None] * len(ws)

        for w, p, l in zip(ws, ps, ls):
            if not p:
                p = [None] * len(w)

            for wi, pi, li in zip(w, p, l):
                self.predictor.update(wi, pi, li)


    def decode(self, ws, ps):
        ys = []

        if not ps:
            ps = [None] * len(ws)

        for w, p in zip(ws, ps):
            if not p:
                p = [None] * len(w)

            y = []
            for wi, pi in zip(w, p):
                yi = self.predictor.predict(wi, pi)
                y.append(yi)

            ys.append(y)

        return ys
