import json
import inspect
from collections import OrderedDict
from fate_torch import nn
from fate_torch.base import Sequential
from fate_torch import init
from federatedml.util import LOGGER


def recover_layer_from_json(nn_define, nn_dict):

    if 'layer' not in nn_define:
        raise ValueError('no layer info offered in nn define')
    nn_layer_class = nn_dict[nn_define['layer']]
    init_para_key = inspect.getargspec(nn_layer_class.__init__)[0]
    init_para_key.remove('self')
    param_dict = {}
    for k in init_para_key:
        if k in nn_define:
            param_dict[k] = nn_define[k]

    layer = nn_layer_class(**param_dict)

    if 'initializer' in nn_define:
        if 'weight' in nn_define['initializer']:
            init_para = nn_define['initializer']['weight']
            init_func = init.str_fate_torch_init_func_map[init_para['init_func']]
            init_func(layer, **init_para['param'])

        if 'bias' in nn_define['initializer']:
            init_para = nn_define['initializer']['bias']
            init_func = init.str_fate_torch_init_func_map[init_para['init_func']]
            init_func(layer, init='bias', **init_para['param'])

    return layer


def recover_sequential_from_json(json_nn_define):
    nn_define_dict = json.loads(json_nn_define)
    add_dict = OrderedDict()
    nn_dict = dict(inspect.getmembers(nn))
    for k, v in nn_define_dict.items():
        layer = recover_layer_from_json(v, nn_dict)
        add_dict[k] = layer

    return Sequential(add_dict)

