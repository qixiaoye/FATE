from torch import optim

from fate_torch.base import FateTorchOptimizer
class ASGD(FateTorchOptimizer):
        
    def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['lambd'] = lambd
        self.param_dict['alpha'] = alpha
        self.param_dict['t0'] = t0
        self.param_dict['weight_decay'] = weight_decay
        # optim.ASGD.__init__(self, **self.param_dict)
    
    
class Adadelta(FateTorchOptimizer):
        
    def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['rho'] = rho
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        # optim.Adadelta.__init__(self, **self.param_dict)
    
    
class Adagrad(FateTorchOptimizer):
        
    def __init__(self, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['lr_decay'] = lr_decay
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['initial_accumulator_value'] = initial_accumulator_value
        self.param_dict['eps'] = eps
        # optim.Adagrad.__init__(self, **self.param_dict)
    
    
class Adam(FateTorchOptimizer):
        
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['amsgrad'] = amsgrad
        # optim.Adam.__init__(self, **self.param_dict)
    
    
class AdamW(FateTorchOptimizer):
        
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['amsgrad'] = amsgrad
        # optim.AdamW.__init__(self, **self.param_dict)
    
    
class Adamax(FateTorchOptimizer):
        
    def __init__(self, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        # optim.Adamax.__init__(self, **self.param_dict)
    
    
class LBFGS(FateTorchOptimizer):
        
    def __init__(self, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['max_iter'] = max_iter
        self.param_dict['max_eval'] = max_eval
        self.param_dict['tolerance_grad'] = tolerance_grad
        self.param_dict['tolerance_change'] = tolerance_change
        self.param_dict['history_size'] = history_size
        self.param_dict['line_search_fn'] = line_search_fn
        # optim.LBFGS.__init__(self, **self.param_dict)
    
    
class RMSprop(FateTorchOptimizer):
        
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['alpha'] = alpha
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['momentum'] = momentum
        self.param_dict['centered'] = centered
        # optim.RMSprop.__init__(self, **self.param_dict)
    
    
class Rprop(FateTorchOptimizer):
        
    def __init__(self, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50), **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['etas'] = etas
        self.param_dict['step_sizes'] = step_sizes
        # optim.Rprop.__init__(self, **self.param_dict)
    
    
class SGD(FateTorchOptimizer):
        
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['momentum'] = momentum
        self.param_dict['dampening'] = dampening
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['nesterov'] = nesterov
        # optim.SGD.__init__(self, **self.param_dict)
    
    
class SparseAdam(FateTorchOptimizer):
        
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, **kwargs):
        FateTorchOptimizer.__init__(self)
        self.param_dict.update(kwargs)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        # optim.SparseAdam.__init__(self, **self.param_dict)
    
    