
from matrix_calculus import *
from matrix_calculus.matrix_massage import massage2canonical
from matrix_calculus.show_latex import show_latex

def main():
    A = Variable("A")
    B = Variable("B")
    C = Variable("C")
    D = Variable("D")
    Y = Variable("Y")
    X = Variable("X")

    expr = Tr(A*X*B)
    wrt = X
    print "Jacobian"
    dX = show_derivation(expr,wrt)
    show_latex(dX,expr,wrt)

    expr = Tr(A*X.T*B*X*C)
    wrt = X
    print "Jacobian"
    dX = show_derivation(expr,wrt)
    show_latex(dX,expr,wrt)

    expr = Tr((Y-D*X).T*(Y-D*X))
    wrt = D
    print "Jacobian"
    dX = show_derivation(expr,wrt)
    show_latex(dX,expr,wrt)

    print "Hessian"
    dX = show_derivation(expr,wrt,hessian=True)
    show_latex(dX,expr,wrt)
    show_latex(dX,expr,wrt,hessian=True)

def show_derivation(expr,wrt,hessian=False):
    print "{} = ".format(DifferentialExpr(expr,wrt))
    dX = d(expr,wrt,hessian=hessian)
    print dX
    print "-->"
    dX = massage2canonical(dX)
    print dX
    print ""
    return dX

if __name__ == '__main__':
    main()
