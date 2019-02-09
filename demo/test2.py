

from matrix_calculus import *
from matrix_calculus.matrix_massage import massage2canonical
from matrix_calculus.show_latex import show_latex


def main():
    W = Variable("W")
    z = Variable("z")

    u = W * z
    v = W.T * u

    expr = Tr(u.T * u * (v.T * v).I)
    wrt = X
    print("Jacobian")
    dX = show_derivation(expr, wrt)
    show_latex(dX, expr, wrt)


def show_derivation(expr, wrt, hessian=False):
    print("{} = ".format(DifferentialExpr(expr, wrt)))
    dX = d(expr, wrt, hessian=hessian)
    print(dX)
    print("-->")
    dX = massage2canonical(dX)
    print(dX)
    print("")
    return dX


if __name__ == '__main__':
    main()
