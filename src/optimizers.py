class DeleteGradient(object):
    name = 'DeleteGradient'

    def __init__(self, target_params):
        self.target_params = target_params


    def __call__(self, optimizer):
        for name, param in optimizer.target.namedparams():
            for tgt in self.target_params:
            # for tgt, idx in self.target_params.items():
                if tgt in name:
                    param.grad = None

                    # if idx < 0:
                    #     param.grad = None
                    # else:
                    #     self.delete_gradient(param, idx)


    # def delete_gradient(self, param, index):
    #     xp = cuda.get_array_module(param)
    #     dim = param.shape[1]
    #     print(param.grad[index:5])
    #     param.grad[index] = xp.zeros(dim, dtype='f')
    #     print(param.grad[index:5])
        

