import json
from torch.nn import Sequential as tSequential


class FateTorchLayer(object):

    def __init__(self):
        self.param_dict = dict()
        self.initializer = {'weight': None, 'bias': None}

    def to_json(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['layer'] = type(self).__name__
        ret_dict['initializer'] = {}
        if self.initializer['weight']:
            ret_dict['initializer']['weight'] = self.initializer['weight']
        if self.initializer['bias']:
            ret_dict['initializer']['bias'] = self.initializer['bias']
        return ret_dict


class FateTorchLoss(object):

    def __init__(self):
        self.param_dict = {}

    def to_json(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['loss_fn'] = type(self).__name__
        return ret_dict


class FateTorchOptimizer(object):

    def __init__(self):
        self.param_dict = dict()

    def to_json(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['optimizer'] = type(self).__name__
        return self.param_dict


class Sequential(tSequential):

    def to_dict(self):
        """
        get the structure of current sequential
        """
        rs = {}
        for k in self._modules:
            rs[k] = self._modules[k].to_json()
        return rs

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def add(self, layer):
        if isinstance(layer, Sequential):
            self._modules = layer._modules
        elif isinstance(layer, FateTorchLayer):
            self.add_module(str(len(self)), layer)
        else:
            raise ValueError('unknown input layer type {}, this type is not supported'.format(type(layer)))

    @staticmethod
    def get_loss_config(loss: FateTorchLoss):
        return loss.to_json()

    @staticmethod
    def get_optimizer_config(optimizer: FateTorchOptimizer):
        return optimizer.to_json()

    def get_network_config(self):
        return self.to_dict()







