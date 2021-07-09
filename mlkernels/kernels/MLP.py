import lab as B
from algebra.util import identical
from matrix import Dense
import numpy as np
import torch
import lab.torch
from mlkernels import Kernel, pairwise, elwise
from . import _dispatch
from .. import Kernel

four_over_tau = torch.tensor(2./B.pi)

__all__ = ["MLP"] 

class MLP(Kernel):
    """MultiLayer Perceptron Kernel.

    Args:
        input_dim - int
        variance - float
        bias_variance - float
        weight_variance - float
    """

    def __init__(self, input_dim, variance=1., weight_variance=1., 
    bias_variance=1., ARD=False):
        self.variance = variance
        self.ARD = ARD
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        if self.ARD:
            assert self.weight_variance.shape[0] == input_dim
   
    def _comp_prod(self, X, X2=None):
        if X2 is None:
            return ((X**2)*self.weight_variance).sum(axis=1)+self.bias_variance
        else:
            return (X*self.weight_variance)@(X2.T)+self.bias_variance

    def _compute(self, X, X2=None):
        self.X = X
        if X2 is None:
            # X = torch.tensor(X) # It should be tensor beforehand
            X_denom = B.sqrt(self._comp_prod(X)+1.)
            X2_denom = X_denom
            X2 = X
        
        else:
            # X = torch.tensor(X)
            # X2 = torch.tensor(X2)
            X_denom = B.sqrt(self._comp_prod(X)+1.)
            X2_denom = B.sqrt(self._comp_prod(X2)+1.)
        XTX = self._comp_prod(X,X2)/X_denom[:,None]/X2_denom[None,:]
        print(self.weight_variance)
        return self.variance*four_over_tau*B.arcsin(XTX)

    def render(self, formatter):
        # This method determines how the kernel is displayed.
        return f"neural_network({formatter(self.__init__)})"

    @property
    def _stationary(self):
       return False

    @_dispatch
    def __eq__(self, other: "MLP"):
        return True

@_dispatch
def pairwise(k: MLP, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(x, y))


@_dispatch
def elwise(k: MLP, x: B.Numeric, y: B.Numeric):
    return k._compute(x, y)








