""""
A module for doing simple matrix calculus in Python.

An example of computing d(||Y-DX||_F^2)
>>> d("||Y-DX||_F^2","X")
Tr(2(Y-DX)^T DdX)
"""

"""
Basic rules:

dA = 0
d(aX) = adX
d(X+Y) = dX + dY
d(tr(X)) = tr(dX)
d(XY) = (dX)Y + XdY

d(X \kron Y) = (dX)  \kron Y + X \kron dY
d(X \circ Y) = (dX)  \circ Y + X \circ dY
dX^-1 = -X^-1 (dX) X^-1
d|X| = |X|tr(X^-1 dX)
dlog|X| = tr(X^-1 dX)
dX* = (dX)*

*: any operator that rearranges elements

Source: T.P. Minka, "Old and New Matrix Algebra Useful for Statistics", 2000.

@author Tommi Kerola

"""

import copy
from matrix_expr import *

def d(expr,wrt,hessian=False):
  """
  Differential operator.
  """
  if type(expr) == str:
    expr = Expr().from_string(expr)
  if type(wrt) == str:
    wrt = Variable(wrt)
  if hessian:
    # 1. Calculate differential.
    expr = d(expr,wrt)
    # 2. Calculate differential of differential.
    expr.make_dx_constant(wrt)
    return d(expr,wrt)

  expr = copy.deepcopy(expr)
  prev_expr = copy.deepcopy(expr)

  if (isinstance(expr, Variable) and expr.name != wrt.name) or isinstance(expr,Scalar):
    # dA = 0
    expr = NullExpr()
    #expr = Scalar(0)
  elif (isinstance(expr, Variable) and expr.name == wrt.name):
    # dX
    expr = DifferentialExpr(expr,wrt)
  elif isinstance(expr, DifferentialExpr):
    # d(dX) = 0
    expr = NullExpr()
  elif isinstance(expr, ScalarMulExpr):
    # d(aX) = adX
    expr = expr.children[0]*d(expr.children[1],wrt)
  elif isinstance(expr, AddExpr):
    # d(X+Y) = dX + dY
    expr = d(expr.children[0],wrt) + d(expr.children[1],wrt)
  elif isinstance(expr, SubExpr):
    # d(X+Y) = dX - dY
    expr = d(expr.children[0],wrt) - d(expr.children[1],wrt)
  elif isinstance(expr, TraceExpr):
    # d(tr(X)) = tr(dX)
    expr = TraceExpr(d(expr.children[0],wrt))
  elif isinstance(expr, MatMulExpr):
    # d(XY) = (dX)Y + XdY
    expr = d(expr.children[0],wrt)*expr.children[1] + expr.children[0]*d(expr.children[1],wrt)
  elif isinstance(expr, StarExpr):
    # dX* = (dX)*
    expr = copy.copy(expr)
    expr.children[0] = d(expr.children[0],wrt)
    if isinstance(expr.children[0], NullExpr):
      expr = NullExpr()
  else:
    expr = DifferentialExpr(expr,wrt) # In this case, we do not know how to go further
    print "Warning: Don't know how to process {}".format(expr)

  #print "d{} -> {}".format(prev_expr,expr)

  return expr
