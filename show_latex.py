
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)
plt.rc('figure', facecolor='white')

def show_latex(expr_grad,expr_orig=None,wrt=None,hessian=False):
  """
  Displays a gradient expression in LaTeX using Matplotlib.

  Keyword args:
  - expr_grad: Gradient to display.
  - expr_orig: Expression before taking the gradient (optional).
  - wrt: The variable the gradient is taken with respect to (optional).
  - hessian:  Whether the gradient is a Hessian matrix (optional).
  """

  latex_str = expr_grad.toLatex()
  if not expr_orig is None and not wrt is None:
    if hessian:
      wrt_latex_str = r"\partial{{{}}} \partial{{{}}}".format(wrt.toLatex(),wrt.T.toLatex())
    else:
      wrt_latex_str = r"\partial{{{}}}".format(wrt.toLatex())
    latex_str = r"\frac{{ \partial{{{}}} }}{{ {} }} = ".format(expr_orig.toLatex(),wrt_latex_str) + latex_str
  #latex_str = r"$\frac{ \partial\|\mathbf{Y}-\mathbf{D}\mathbf{X}\|_2^2 }{\partial \mathbf{D}} = \mathbf{X} (\mathbf{Y}-\mathbf{D}}\mathbf{X})^T \partial \mathbf{D}$"
  plt.figtext(0.5,0.5,"${}$".format(latex_str),horizontalalignment='center')

  plt.show()
