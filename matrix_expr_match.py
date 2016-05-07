"""
Utilty functions for matching matrix expressions with gradient rules.
"""

from collections import defaultdict
from matrix_expr import *

def peel(expr):
  return expr.children

def match_deepest(expr,cases):
  matches = []
  for case in cases:
    case_expr_matches = {}
    try:
      match_case(expr,case,case_expr_matches)
    except MatchError:
      continue
    case_vars = get_var_names(case)
    if set(case_expr_matches.keys()) == case_vars:
      matches.append(case)
  matches.sort(key=lambda x: -len(x)) # Put deepest case first
  return matches

def match_case(expr,case,d):
  if type(case) == Variable:
    # expr can contain anything
    if case.name in d:
      if d[case.name] != expr:
        raise MatchError("Variable {} cannot correspond to both {} and {}.".format(case.name,d[case.name],expr))
    else:
      d[case.name] = expr
  elif type(case) == ScalarVariable:
    # expr can contain a scalar
    if type(expr) == Scalar or type(expr) == ScalarVariable:
      if case.name in d:
        if d[case.name] != expr:
          raise MatchError("Variable {} cannot correspond to both {} and {}.".format(case.name,d[case.name],expr))
      else:
        d[case.name] = expr
  elif type(case) == DifferentialExpr:
    if expr.contains(DifferentialExpr):
      # Throw away the DifferentialExpr in the case
      # Re-evaluate expr at the next level
      match_case(expr,case.children[0],d)
    else:
      raise MatchError("No match at differential operator.")
  elif type(case) == type(expr):
    # The structure of the case and expr matches, look at subcases.
    subcases = peel(case)
    subexprs = peel(expr)
    # This case is matching if all subcases are matching
    dsubs = []
    for subexpr,subcase in zip(subexprs,subcases):
      dsub = {}
      match_case(subexpr,subcase,dsub)
      case_vars = get_var_names(subcase)
      # Make sure all case keys were set
      for varname in case_vars:
        if varname in d:
          if not varname in dsub:
            raise MatchError("Match in one child, but not the other.")
          elif d[varname] != dsub[varname]:
            raise MatchError("Vars in subexpressions matches to different expressions.")
      d.update(dsub)
    #ok = True
    #for i in range(len(dsubs)):
    #  for j in range(i+1,len(dsubs)):
    #    if set(dsubs[i].keys()) == set(dsubs[j].keys()):
    #      if set(dsubs[i].values()) != set(dsubs[j].values()):
    #        ok = False
    #if ok:
    #  for dsub in dsubs:
    #    d.update(dsub)
      

def translate_case(expr,start_case,end_case):
  """
  Translates expr according to the pattern
  start case -> end_case

  E.g.
  Tr(X*A) -> Tr(A*X)

  Raises:
   - MatchError: If expr and start_case does not match.
  """
  end_case = copy.deepcopy(end_case)
  # Get a list of all the variables in start_case and end_case.
  # These must be same
  start_var_parent_dict = defaultdict(list)
  end_var_parent_dict = defaultdict(list)
  create_vars_parent_dict(start_case,start_var_parent_dict)
  create_vars_parent_dict(end_case,end_var_parent_dict)
  #print ""
  #print start_var_parent_dict
  #print end_var_parent_dict

  if start_var_parent_dict.keys() != end_var_parent_dict.keys():
    raise ValueError("Start case and end case must be contain the same variables.")

  case_expr_matches = {}
  match_case(expr,start_case,case_expr_matches)
  #print start_case,"->",end_case
  #print case_expr_matches

  for varname in end_var_parent_dict.iterkeys():
    for (child_index,parent) in end_var_parent_dict[varname]:
      subexpr = case_expr_matches[varname]
      if isinstance(parent,TransposeExpr) and isinstance(subexpr,TransposeExpr):
        # A bit of a hack for avoiding X'', and returning X instead.
        i,p = get_parent(parent,end_case)
        p.children[i] = subexpr.children[0]
      else:
        parent.children[child_index] = subexpr
  return end_case

def create_vars_parent_dict(expr,d):
  for child_index,child in enumerate(expr.children):
    if isinstance(child,Variable):
      # Map variable to its parent
      d[child.name].append((child_index,expr))
    else:
      create_vars_parent_dict(child,d)

def get_var_names(expr):
  nameset = set()
  get_var_names_(expr,nameset)
  return nameset

def get_var_names_(expr,nameset):
  if isinstance(expr,Variable) or isinstance(expr,ScalarVariable):
    nameset.add(expr.name)
  for child in expr.children:
    get_var_names_(child,nameset)

def get_parent(expr,parent):
  for child_index,child in enumerate(parent.children):
    if id(expr) == id(child):
      return child_index,parent
    else:
      child_child_index,child_parent = get_parent(expr,child)
      if not child_parent is None:
        return child_child_index,child_parent
  return -1,None
      

class MatchError(Exception):
  pass
