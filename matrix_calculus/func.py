
import numpy as np

def expr2func(expr,wrt,const_dict,wrt_shape=None):
    """
    Transforms an Expr to a functon
    of the wrt Variable.
    """
    def f(x):
        if wrt_shape is not None:
            X = np.reshape(x,wrt_shape)
        return expr.eval(X,wrt,const_dict)
    return f

