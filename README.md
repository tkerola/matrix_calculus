
Matrix Calculus
===============
A simple library for calculating matrix derivatives.

Usage
=====
```
from matrix_calculus import *
from matrix_massage import massage2canonical
A = Variable("A")
B = Variable("B")
X = Variable("X")

expr = Tr(A*X*B)
wrt = X
dX = d(expr,wrt,hessian=hessian)
print dX # Raw form
dX = massage2canonical(dX)
print dX # Canonical form
```

The canonical form can consequently be directly used in an optimization library such as [CVXPY](http://www.cvxpy.org/en/latest/).

See also `test.py`.

References
==========
Based on
Minka, Thomas P. "Old and new matrix algebra useful for statistics." See www. stat. cmu. edu/minka/papers/matrix. html (2000).

License
=======
MIT License.
