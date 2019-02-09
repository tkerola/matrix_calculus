
import numpy as np


def expr2func(expr, wrt, const_dict, wrt_shape=None, res_shape=None, is_grad=False):
    """
    Transforms an Expr to a functon
    of the wrt Variable.
    """
    def f(x):
        if wrt_shape is not None:
            x = np.reshape(x, wrt_shape)
        y = expr.eval(x, wrt, const_dict, is_grad)
        if res_shape is not None:
            y = np.reshape(y, res_shape)
        return y
    return f
