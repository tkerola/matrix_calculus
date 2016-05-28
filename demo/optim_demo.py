"""
Optimization demo.
"""

"""
Suppose we have matrices Y,D and want to solve
    argmin_X 0.5 || Y - DX ||_2^2         (1)
for X but don't know how.
"""

# Let's first generate some noisy data.
import numpy as np
np.random.seed(42)
p,n = 2,5
X_true = np.random.random((p,n))
sigma = 1e-3
D = np.diag(np.random.normal(0,sigma,size=(p)))
Y = D.dot(X_true)

# We can solve the optimization problem by using gradient descent.
# Support we don't know the gradient of (1).
# We can use matrix_calculus for this.
from matrix_calculus import *
from matrix_calculus.matrix_massage import massage2canonical
vY = Variable("Y")
vD = Variable("D")
vX = Variable("X")
# (1) can be rewritten as 0.5Tr((Y-DX)^T (Y-DX))
expr = 0.5*Tr((vY-vD*vX).T*(vY-vD*vX))
wrt = vX
dX = d(expr,wrt)
print "Jacobian:"
print dX
# The derivative is 0.5Tr((Dd(X))'(Y-DX)+(Y-DX)'Dd(X)),
# which is correct, but we need the canonical form
# of the derivative in order to to gradient descent.
# We can get this form by using massage2canonical.
dX = massage2canonical(dX,verbose=False)
print "Jacobian (canonical):"
print dX
# The (canonical) derivative is 0.5Tr(((D'(Y-DX))'+(Y-DX)'D)d(X)).
# We can solve this using L-BFGS
from scipy.optimize import fmin_l_bfgs_b
X0=np.random.random((p,n))
def f_true(x):
    X = x.reshape((p,n))
    return 0.5*np.sum((Y-D.dot(X))**2)
def fp_true(x):
    X = x.reshape((p,n))
    return 0.5*((D.T.dot(Y-D.dot(X))).T+(Y-D.dot(X)).T.dot(D)).ravel()
from matrix_calculus.func import expr2func
const_dict = {'Y':Y,'D':D}
f = expr2func(expr,wrt,const_dict,wrt_shape=(p,n))
#fp = lambda x: expr2func(dX,wrt,const_dict,wrt_shape=(p,n))(x).T.ravel()
def fp(x):
    X = x.reshape((p,n))
    return dX.eval(X,wrt,const_dict,is_grad=True).ravel()
print np.linalg.norm(f(X0) - f_true(X0))
print np.linalg.norm(fp(X0) - fp_true(X0))

x,min_val,d = fmin_l_bfgs_b(f,X0,fp)
X = x.reshape((p,n))
for k in ['warnflag','funcalls','nit','grad']:
    print "{}: {}".format(k,d[k])
print "x: {}".format(x)
