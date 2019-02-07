"""
Matrix expression classes.

@author Tommi Kerola

"""

import copy

import numpy as np

class Expr(object):
  def __init__(self,precedence_level):
    """
    Matrix expression base class.

    Keyword args:
      - precedence_level: The level of precedence of this expression.

          Expressions with lower precedence level will be evaluated
          before expressions of higher level.

          Expressions of equal level will be evaluated from left to right.

          The below table follows the convention taken in C++:
          http://en.cppreference.com/w/c/language/operator_precedence

          Precedence level  Operator
          0                 DifferentialExpr,TraceExpr
          1                 Variable, Scalar, NullExpr, StarExpr, TransposeExpr
          2                 unary plus and minus
          3                 MatMulExpr
          4                 AddExpr, SubExpr
    """
    super(Expr,self).__init__()
    self.children = []
    self.precedence_level = precedence_level
  def __hash__(self):
    raise NotImplementedError()
  def from_string(self,s):
    pass
  def contains(self,expr_type):
    if isinstance(self,expr_type):
      return True
    else:
      return any(map(lambda c: c.contains(expr_type), self.children))
  def make_dx_constant(self,wrt):
    for i,child in enumerate(self.children):
      if isinstance(child,DifferentialExpr) and child.children[0] == wrt:
        const_child = copy.deepcopy(wrt)
        const_child.name = str(child)
        self.children[i] = const_child
      else:
        child.make_dx_constant(wrt)
  def toLatex(self):
    return ""

  def eval(self,x,wrt,const_dict,is_grad=False):
    raise NotImplementedError
  def __repr__(self):
    return self.__str__()
  def __str__(self):
    return ""
  def __len__(self):
    return 1 + sum(map(len, self.children))
  def __eq__(self,other):
    if type(self) != type(other):
      return False
    return self.children == other.children
  def __ne__(self,other):
    return not (self == other)

  def __add__(self,other):
    if type(self) == NullExpr:
      return other
    elif type(other) == NullExpr:
      return self
    elif self == other:
      return ScalarMulExpr(Scalar(2),self)
    else:
      return AddExpr(self,other)
  def __sub__(self,other):
    if type(self) == NullExpr:
      return other
    elif type(other) == NullExpr:
      return self
    elif self == other:
      return NullExpr()
    else:
      return SubExpr(self,other)
  def __mul__(self,other):
    if type(self) == NullExpr or type(other) == NullExpr:
      return NullExpr()
    elif type(self) == Scalar and type(other) == Scalar:
      return Scalar(self.value*other.value)
    elif type(self) == Scalar:
      return ScalarMulExpr(self,other)
    elif type(other) == Scalar:
      return ScalarMulExpr(other,self)
    elif type(self) == ScalarVariable:
      return ScalarMulExpr(self,other)
    elif type(self) == ScalarMulExpr:
      return ScalarMulExpr(self.children[0],self.children[1]*other)
    elif type(other) == ScalarMulExpr:
      return ScalarMulExpr(other.children[0],self*other.children[1])
    elif type(self) == ScalarMulExpr and type(other) == ScalarMulExpr:
      return ScalarMulExpr(self.children[0]*other.children[0],self.children[1]*other.children[1])
    else:
      return MatMulExpr(self,other)
  def __rmul__(self,other):
    # other * self
    if isinstance(other,float) or isinstance(other,int):
      return Scalar(other)*self
    raise NotImplementedError
  def __floordiv__(self,other):
    raise NotImplementedError
  def __div__(self,other):
    raise NotImplementedError
  def __truediv__(self,other):
    raise NotImplementedError
  def __pow__(self,other):
    raise NotImplementedError
  def __neg__(self):
    raise NotImplementedError
  def __pos__(self):
    return self
  T = property(lambda self: self.children[0] if isinstance(self,TransposeExpr) else TransposeExpr(self))


class DifferentialExpr(Expr):
  def __init__(self,expr,wrt):
    super(DifferentialExpr,self).__init__(0)
    self.children = [expr]
    self.wrt = wrt
  def __hash__(self):
    h = hash(self.wrt)
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    return 1.
  def __str__(self):
    brackets = self.precedence_level < self.children[0].precedence_level
    return "d{}{}{}".format("(" if brackets else "",self.children[0],")" if brackets else "")
  def __eq__(self,other):
    return self.children == other.children and self.wrt == other.wrt
  def toLatex(self):
    return r"\partial{{{}}}".format(self.children[0].toLatex())


class Variable(Expr):
  reserved_names = {
      "T",
      }
  def __init__(self,name):
    super(Variable,self).__init__(1)
    if name in self.reserved_names:
      raise ValueError("Cannot create variable. \"{}\" is a reserved name.".format(name))
    self.name = name
  def __hash__(self):
    return hash(self.name)
  def eval(self,x,wrt,const_dict,is_grad=False):
    return x if wrt.name == self.name else const_dict[self.name]
  def __str__(self):
    return self.name
  def __eq__(self,other):
    if type(self) != type(other):
      return False
    return self.name == other.name
  def toLatex(self):
    return r"\mathbf{{{}}}".format(self.name)


class ScalarVariable(Expr):
  reserved_names = {
      }
  def __init__(self,name):
    super(ScalarVariable,self).__init__(1)
    if name in self.reserved_names:
      raise ValueError("Cannot create variable. \"{}\" is a reserved name.".format(name))
    self.name = name
  def __hash__(self):
    return hash(self.name)
  def eval(self,x,wrt,const_dict,is_grad=False):
    return x if wrt.name == self.name else const_dict[self.name]
  def __str__(self):
    return self.name
  def __eq__(self,other):
    if type(self) != type(other):
      return False
    return self.name == other.name
  def toLatex(self):
    return r"{{{}}}".format(self.name)


class Scalar(Expr):
  def __init__(self,value):
    super(Scalar,self).__init__(1)
    self.value = value
  def __hash__(self):
    return hash(self.value)
  def eval(self,x,wrt,const_dict,is_grad=False):
    return self.value
  def __str__(self):
    return "{}".format(self.value)
  def __eq__(self,other):
    return self.value == other.value
  def toLatex(self):
    return r"{{{}}}".format(self.value)


class NullExpr(Expr):
  def __init__(self):
    super(NullExpr,self).__init__(1)
  def eval(self,x,wrt,const_dict,is_grad=False):
    return 0.
  def __hash__(self):
    return hash('null')
  def __str__(self):
    return "0"
  def __eq__(self,other):
    return type(self) == type(other)


class AddExpr(Expr):
  def __init__(self,left,right):
    super(AddExpr,self).__init__(4)
    self.children = [left,right]
  def __hash__(self):
    h = hash('+')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    return self.children[0].eval(x,wrt,const_dict,is_grad) + self.children[1].eval(x,wrt,const_dict,is_grad)
  def __str__(self):
    return "{}+{}".format(self.children[0],self.children[1])
  def toLatex(self):
    return r"{{{}}}+{{{}}}".format(self.children[0].toLatex(),self.children[1].toLatex())


class SubExpr(Expr):
  def __init__(self,left,right):
    super(SubExpr,self).__init__(4)
    self.children = [left,right]
  def __hash__(self):
    h = hash('-')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    return self.children[0].eval(x,wrt,const_dict,is_grad) - self.children[1].eval(x,wrt,const_dict,is_grad)
  def __str__(self):
    return "{}-{}".format(self.children[0],self.children[1])
  def toLatex(self):
    return r"{{{}}}-{{{}}}".format(self.children[0].toLatex(),self.children[1].toLatex())


class ScalarMulExpr(Expr):
  def __init__(self,left,right):
    super(ScalarMulExpr,self).__init__(3)
    self.children = [left,right]
  def __hash__(self):
    h = hash('*')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    return np.dot(self.children[0].eval(x,wrt,const_dict,is_grad), self.children[1].eval(x,wrt,const_dict,is_grad))
  def __str__(self):
    left_brackets = self.precedence_level < self.children[0].precedence_level
    right_brackets = self.precedence_level < self.children[1].precedence_level
    if self.children[0] == Scalar(1):
      return "{}{}{}".format(
          "(" if right_brackets else "",self.children[1],")" if right_brackets else "")
    else:
      return "{}{}{}{}{}{}".format(
          "(" if left_brackets else "",self.children[0],")" if left_brackets else "",
          "(" if right_brackets else "",self.children[1],")" if right_brackets else "")
  def toLatex(self):
    left_brackets = self.precedence_level < self.children[0].precedence_level
    right_brackets = self.precedence_level < self.children[1].precedence_level
    if self.children[0] == Scalar(1):
      return "{}{}{}".format(
          "(" if right_brackets else "",self.children[1].toLatex(),")" if right_brackets else "")
    else:
      return r"{}{}{}{}{}{}".format(
          "(" if left_brackets else "",self.children[0].toLatex(),")" if left_brackets else "",
          "(" if right_brackets else "",self.children[1].toLatex(),")" if right_brackets else "")


class MatMulExpr(Expr):
  def __init__(self,left,right):
    super(MatMulExpr,self).__init__(3)
    self.children = [left,right]
  def __hash__(self):
    h = hash('@')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    return np.dot(self.children[0].eval(x,wrt,const_dict,is_grad), self.children[1].eval(x,wrt,const_dict,is_grad))
  def __str__(self):
    left_brackets = self.precedence_level < self.children[0].precedence_level
    right_brackets = self.precedence_level < self.children[1].precedence_level
    return "{}{}{}{}{}{}".format(
        "(" if left_brackets else "",self.children[0],")" if left_brackets else "",
        "(" if right_brackets else "",self.children[1],")" if right_brackets else "")
  def toLatex(self):
    left_brackets = self.precedence_level < self.children[0].precedence_level
    right_brackets = self.precedence_level < self.children[1].precedence_level
    return r"{}{}{}{}{}{}".format(
        "(" if left_brackets else "",self.children[0].toLatex(),")" if left_brackets else "",
        "(" if right_brackets else "",self.children[1].toLatex(),")" if right_brackets else "")


class TraceExpr(Expr):
  def __init__(self,expr):
    super(TraceExpr,self).__init__(0)
    self.children = [expr]
  def __hash__(self):
    h = hash('Tr')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad=False):
    if is_grad:
        return self.children[0].eval(x,wrt,const_dict,is_grad)
    else:
        return np.trace(self.children[0].eval(x,wrt,const_dict,is_grad))
  def __str__(self):
    return "Tr({})".format(self.children[0])
  def toLatex(self):
    return r"\mathrm{{Tr}}({})".format(self.children[0].toLatex())


class StarExpr(Expr): # Any operator that rearranges elements
  def __init__(self,expr,symbol):
    super(StarExpr,self).__init__(1)
    self.children = [expr]
    self.symbol = symbol
  def __hash__(self):
    h = hash('star')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad):
    raise NotImplementedError
  def __str__(self):
    brackets = self.precedence_level < self.children[0].precedence_level
    return "{}{}{}{}".format("(" if brackets else "",self.children[0],")" if brackets else "",self.symbol)
  def toLatex(self):
    brackets = self.precedence_level < self.children[0].precedence_level
    return r"{{{}{}{}}}{}".format("(" if brackets else "",self.children[0].toLatex(),")" if brackets else "",self.symbol)


class TransposeExpr(StarExpr):
  def __init__(self,expr):
    super(TransposeExpr,self).__init__(expr,"'")
  def __hash__(self):
    h = hash('T')
    for c in self.children:
        h *= hash(c)
    return h
  def eval(self,x,wrt,const_dict,is_grad):
    return np.transpose(self.children[0].eval(x,wrt,const_dict,is_grad))


def Tr(expr):
  return TraceExpr(expr)

def print_structure(expr):
  pipe_dict = {0: " "}
  qs = [(0,pipe_dict,expr)]
  indent = 4
  end_tree_chr = "+"
  while len(qs) > 0:
    level,pipe_dict,e = qs.pop() # Get last item
    num_children = len(e.children)
    #print "{:>20} | {}{}".format(e," "*level*indent,e.__class__.__name__)
    if level > 0:
      indent_str = ""
      for level_index in range(level):
        pipe_str = pipe_dict[level_index]
        indent_str += "{:>{}}".format(pipe_str,indent)
        if pipe_str == end_tree_chr:
          pipe_dict[level_index] = " "
      #indent_str += "{:>{}}".format("-",indent)
      indent_str += "-"*(indent-2)
    else:
      indent_str = ""
    print("{}{} = {}".format(indent_str,e.__class__.__name__,e))

    [qs.append((level+1,dict(pipe_dict.items() + {level: "|" if num_children-1-ci < num_children-1 else end_tree_chr}.items()),c)) for ci,c in enumerate(e.children[::-1])]


