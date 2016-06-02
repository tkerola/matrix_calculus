"""

@author Tommi Kerola

"""

import copy
import functools
from matrix_expr import *
from matrix_expr_match import match_deepest, translate_case

CANONICAL_VERBOSE = True

def massage2canonical(expr,verbose=True):
  """
  Massages the given expression
  to canonical form with the dX
  at the top level, if possible.

  An expression is in canonical form if it is written as Tr(AdX),
  so that dX is on the right side.

  For each non-canonical expression, expand only the branch that contains a dX.
  For each canonical expression, combine it with other canonical expressions.
  """
  global CANONICAL_VERBOSE
  CANONICAL_VERBOSE = verbose

  d = lambda e: DifferentialExpr(e,X)

  X = Variable("X")
  A = Variable("A")
  B = Variable("B")
  C = Variable("C")
  s = ScalarVariable("s")
  u = ScalarVariable("u")
  # In all cases, d(X) represents an expression that contains
  # a differential, and is only used for matching.
  # The right-hand side of each case does not contain this differential.
  cases = {
    Tr(d(A)*B) : Tr(B*A),
    d(Tr(A+B)) : Tr(A) + Tr(B),
    d(Tr(A-B)) : Tr(A) - Tr(B),
    Tr(A*d(B).T) : Tr(A.T*B),
    Tr(A*d(C)) + Tr(B*d(C)): Tr((A+B)*C),
    (A*B.T).T + B*A.T: 2*(B*A.T),
    (A.T*B).T + B.T*A: 2*(B.T*A),
    d(A*B).T : B.T*A.T,
    X*(A*X).T + (A*X*X.T).T: 2*(X*X.T*A.T),
    d(A+B)*C : A*C + B*C,
    A*d(B+C) : A*B + A*C,
    d(A-B)*C : A*C - B*C,
    A*d(B-C) : A*B - A*C,
    Tr(A*d(B*C)) : Tr((A*B)*C), # For changing associativity so that B (or C) can eventually move to the right side (use the Tr rule above)
    s*u*Tr(A) : s*Tr(u*A),
    #s*Tr(A) : s*Tr(A),
    #(s*A)*B: s*(A*B),
    #Tr(A) + Tr(B): Tr(A+B),
    #Tr(A) - Tr(B): Tr(A-B),
    #d(A+B): A+B,
    #d(A-B): A-B,
  }


  expr = massage2canonical_stage1(expr,cases,[1])
  #print "after stage1:",expr
  #expr = massage2canonical_stage2(expr)
  return expr

def massage2canonical_stage1(expr,cases,levels):
  expr = copy.deepcopy(expr)
  prev_expr = copy.deepcopy(expr)

  matches = match_deepest(expr,cases.keys())
  if len(matches) > 0:
    best_case = matches[0]
    expr = translate_case(expr,best_case,cases[best_case])
    #print_structure(prev_expr)
    #print_structure(expr)
    new_expr_copy = copy.deepcopy(expr)
    #print "Matches:,matches
    if CANONICAL_VERBOSE:
        print "[{}] Applying {} -> {}".format(".".join(map(str,levels)),best_case,cases[best_case])
        print "[{}] :: {} -> {}".format(".".join(map(str,levels)),prev_expr,new_expr_copy)

    expr.children = [massage2canonical_stage1(child,cases,levels+[1]) for child_index,child in enumerate(expr.children)]
    levels[-1] += 1
    expr = massage2canonical_stage1(expr,cases,levels)
  else:
    #print ":: {} -> No match.".format(prev_expr)
    #print_structure(prev_expr)
    expr.children = [massage2canonical_stage1(child,cases,levels+[child_index+1]) for child_index,child in enumerate(expr.children)]
    pass

  if False:
    if isinstance(expr, AddExpr):
      left, right = expr.children[0], expr.children[1]
      expr = massage2canonical_stage1(left) + massage2canonical_stage1(right)
    elif isinstance(expr, SubExpr):
      left, right = expr.children[0], expr.children[1]
      expr = massage2canonical_stage1(left) - massage2canonical_stage1(right)
    elif isinstance(expr, MatMulExpr):
      left, right = expr.children[0], expr.children[1]
      if left.contains(DifferentialExpr):
        if isinstance(left, AddExpr):
          # (A+B)C = AC + BC
          A,B,C = left.children[0], left.children[1], right
          expr = massage2canonical_stage1(A*C) + massage2canonical_stage1(B*C)
        elif isinstance(left, SubExpr):
          # (A-B)C = AC - BC
          A,B,C = left.children[0], left.children[1], right
          expr = massage2canonical_stage1(A*C) - massage2canonical_stage1(B*C)
        else:
          expr = massage2canonical_stage1(left)*right
      if right.contains(DifferentialExpr):
        if isinstance(right, AddExpr):
          # A(B+C) = AB + AC
          A,B,C = left, right.children[0], right.children[1]
          expr = massage2canonical_stage1(A*B) + massage2canonical_stage1(A*C)
        elif isinstance(right, SubExpr):
          # A(B-C) = AB - AC
          A,B,C = left, right.children[0], right.children[1]
          expr = massage2canonical_stage1(A*B) - massage2canonical_stage1(A*C)
        else:
          expr = left*massage2canonical_stage1(right)
    elif isinstance(expr, ScalarMulExpr):
      left, right = expr.children[0], expr.children[1]
      if right.contains(DifferentialExpr):
        expr = ScalarMulExpr(left,massage2canonical_stage1(right))
    elif isinstance(expr, TransposeExpr):
      child = expr.children[0]
      if isinstance(child, MatMulExpr) and child.contains(DifferentialExpr):
        # (AB)' = B'A'
        A,B = child.children[0], child.children[1]
        expr = massage2canonical_stage1(B.T * A.T)
    elif isinstance(expr, TraceExpr):
      child = expr.children[0]
      if isinstance(child, MatMulExpr):
        # Tr(AB) = Tr(BA)
        A,B = child.children[0], child.children[1]
        if A.contains(DifferentialExpr): # Move to right side
          expr = massage2canonical_stage1(Tr(massage2canonical_stage1(B*massage2canonical_stage1(A))))
        elif B.contains(DifferentialExpr) and not isinstance(B,DifferentialExpr): # Change associativity so that dX becomes on the right-most side of left
          if isinstance(B,MatMulExpr):
            # Tr(ABC) needs to be handled specially due to the associative property
            # A(BC) = (AB)C
            B,C = B.children[0],B.children[1]
            expr = massage2canonical_stage1(Tr(massage2canonical_stage1(A*massage2canonical_stage1(B))*massage2canonical_stage1(C)))
          elif isinstance(B,TransposeExpr):
            # Tr(AB') = Tr(A'B)
            expr = massage2canonical_stage1(Tr(massage2canonical_stage1((A.T)*massage2canonical_stage1(B.T))))
      elif isinstance(child, AddExpr):
        # Tr(A+B) = Tr(A) + Tr(B)
        A,B = child.children[0], child.children[1]
        expr = massage2canonical_stage1(massage2canonical_stage1(Tr(A)) + massage2canonical_stage1(Tr(B)))
      elif isinstance(child, SubExpr):
        # Tr(A-B) = Tr(A) - Tr(B)
        A,B = child.children[0], child.children[1]
        expr = massage2canonical_stage1(massage2canonical_stage1(Tr(A)) - massage2canonical_stage1(Tr(B)))

  #print "{} -> {}".format(prev_expr,expr)

  return expr

#def massage2canonical_stage2(expr):
#  expr = copy.deepcopy(expr)
#  prev_expr = copy.deepcopy(expr)
#
#  if isinstance(expr, AddExpr):
#    left, right = expr.children[0], expr.children[1]
#    if isinstance(left,TraceExpr) and isinstance(right,TraceExpr):
#      # Tr(A) + Tr(B)
#      A,B = left.children[0], right.children[0]
#      if is_canonical(A) and is_canonical(B):
#        A,X1 = A.children[0], A.children[1]
#        B,X2 = B.children[0], B.children[1]
#        if X1 == X2 and isinstance(X1,DifferentialExpr):
#          C = A+B
#          if isinstance(C,ScalarMulExpr):
#            expr = C.children[0] * Tr(C.children[1] * X1)
#          else:
#            expr = Tr(C*X1)
#    #else:
#    #  expr = massage2canonical_stage2(left) + massage2canonical_stage2(right)
#  elif isinstance(expr, SubExpr):
#    left, right = expr.children[0], expr.children[1]
#    if isinstance(left,TraceExpr) and isinstance(right,TraceExpr):
#      # Tr(A) - Tr(B)
#      A,B = left.children[0], right.children[0]
#      if is_canonical(A) and is_canonical(B):
#        A,X1 = A.children[0], A.children[1]
#        B,X2 = B.children[0], B.children[1]
#        if X1 == X2 and isinstance(X1,DifferentialExpr):
#          C = A-B
#          if isinstance(C,ScalarMulExpr):
#            expr = C.children[0] * Tr(C.children[1] * X1)
#          else:
#            expr = Tr(C*X1)
#    #else:
#    #  expr = massage2canonical_stage2(left) - massage2canonical_stage2(right)
#
#  #print "{} -> {}".format(prev_expr,expr)
#
#  return expr

def is_canonical(expr):
  """
  True if the expression is on
  the form AdX
  """
  if isinstance(expr,MatMulExpr):
    left, right = expr.children[0], expr.children[1]
    return isinstance(right,DifferentialExpr)
  else:
    return False

