"""

@author Tommi Kerola

"""

import copy
import functools
from matrix_calculus.matrix_expr import *
from matrix_calculus.matrix_expr_match import match_deepest, translate_case

CANONICAL_VERBOSE = True


def massage2canonical(expr, verbose=True):
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

    def d(e): return DifferentialExpr(e, X)

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
        Tr(d(A)*B): Tr(B*A),
        d(Tr(A+B)): Tr(A) + Tr(B),
        d(Tr(A-B)): Tr(A) - Tr(B),
        Tr(A*d(B).T): Tr(A.T*B),
        Tr(A*d(C)) + Tr(B*d(C)): Tr((A+B)*C),
        (A*B.T).T + B*A.T: 2*(B*A.T),
        (A.T*B).T + B.T*A: 2*(B.T*A),
        d(A*B).T: B.T*A.T,
        X*(A*X).T + (A*X*X.T).T: 2*(X*X.T*A.T),
        d(A+B)*C: A*C + B*C,
        A*d(B+C): A*B + A*C,
        d(A-B)*C: A*C - B*C,
        A*d(B-C): A*B - A*C,
        # For changing associativity so that B (or C) can eventually move to the right side (use the Tr rule above)
        Tr(A*d(B*C)): Tr((A*B)*C),
        s*u*Tr(A): s*Tr(u*A),
        X*s.I: s.I*X,
        A+-B: A-B,
        # s*Tr(A) : s*Tr(A),
        # (s*A)*B: s*(A*B),
        # Tr(A) + Tr(B): Tr(A+B),
        # Tr(A) - Tr(B): Tr(A-B),
        # d(A+B): A+B,
        # d(A-B): A-B,
    }

    print('try 1')
    expr = massage2canonical_stage1(expr, cases, [1], {}, prev_case=True)

    cases = {
        Tr(d(A)*B): Tr(B*A),
        d(Tr(A+B)): Tr(A) + Tr(B),
        d(Tr(A-B)): Tr(A) - Tr(B),
        Tr(A*d(B).T): Tr(A.T*B),
        Tr(A*d(C)) + Tr(B*d(C)): Tr((A+B)*C),
        (A*B.T).T + B*A.T: 2*(B*A.T),
        (A.T*B).T + B.T*A: 2*(B.T*A),
        ((2*A)*B.T).T + B*(2*A.T): 4*(B*A.T),
        ((2*A.T)*B).T + B.T*(2*A): 4*(B.T*A),
        d(A*B).T: B.T*A.T,
        X*(A*X).T + (A*X*X.T).T: 2*(X*X.T*A.T),
        d(A+B)*C: A*C + B*C,
        A*d(B+C): A*B + A*C,
        d(A-B)*C: A*C - B*C,
        A*d(B-C): A*B - A*C,
        # For changing associativity so that B (or C) can eventually move to the right side (use the Tr rule above)
        Tr(A*d(B*C)): Tr((A*B)*C),
        A*(B*C): (A*B)*C,
        s*u*Tr(A): s*Tr(u*A),
        X*s.I: s.I*X,
        (s*X).I: s.I*X.I,
        # ---
        A+-B: A-B,
        -Scalar(1)*A: -A,
        A*(s*B): s*(A*B),
        A*(B*s): s*(A*B),
        Tr(s*A): s*Tr(A),
        A*s: s*A,
        A*(s*B): s*A*B,
        (u*A)*(s*B): u*s*A*B,
        s*A + s*B: s*(A+B),
        s*A - s*B: s*(A-B),
        B*(s*A).T: s*(B*A.T),
        (s*A).T: s*A.T,
        (s*A)*B: s*(A*B),
        Tr(A) + Tr(B): Tr(A+B),
        Tr(A) - Tr(B): Tr(A-B),
        d(A+B): A+B,
        d(A-B): A-B,
    }
    print('try 2')
    expr = massage2canonical_stage1(expr, cases, [1], {}, prev_case=True)
    print('try 3')
    expr = massage2canonical_stage1(expr, cases, [1], {}, prev_case=True)

    # print("after stage1:",expr)
    # expr = massage2canonical_stage2(expr)
    return expr


def fix_structure(expr):
    for i, c in enumerate(expr.children):
        fix_structure(c)

        if isinstance(c, MatMulExpr):
            a, b = c.children
            if isinstance(a, ScalarMulExpr) and isinstance(b, ScalarMulExpr):
                c = ScalarMulExpr(a.children[0] * b.children[0], a.children[1] * b.children[1])
                expr.children[i] = c
            elif isinstance(a, ScalarMulExpr):
                c = ScalarMulExpr(a.children[0], a.children[1] * b)
                expr.children[i] = c
            elif isinstance(b, ScalarMulExpr):
                c = ScalarMulExpr(b.children[0], a * b.children[1])
                expr.children[i] = c


def massage2canonical_stage1(expr, cases, levels, mem, prev_case=None, return_matches=False):
    # print(expr)
    # print(type(expr))

    # if expr in mem:
    #     return mem[expr]

    while True:
        result = [massage2canonical_stage1(
            child, cases, levels+[child_index+1], mem, prev_case, return_matches=True) for child_index, child in enumerate(expr.children)]
        expr.children = [a for a, b in result]
        child_had_matches = any([b for a, b in result])
        fix_structure(expr)
        if not child_had_matches:
            break

    fix_structure(expr)

    had_matches = False
    while True:
        expr = copy.deepcopy(expr)
        prev_expr = copy.deepcopy(expr)
        matches = match_deepest(expr, cases.keys())
        if len(matches) > 0:
            best_case = matches.pop(0)
            if prev_case is not None and len(matches) > 0:
                while cases[best_case] == prev_case:
                    best_case = matches.pop(0)
        if len(matches) > 0:
            had_matches = True

            expr = translate_case(expr, best_case, cases[best_case])
            # print_structure(prev_expr)
            # print_structure(expr)
            new_expr_copy = copy.deepcopy(expr)
            # print "Matches:,matches
            if CANONICAL_VERBOSE:
                print("[{}] Applying {} -> {}".format(".".join(map(str,
                                                                   levels)), best_case, cases[best_case]))
                print("[{}] :: {} -> {}".format(".".join(map(str, levels)),
                                                prev_expr, new_expr_copy))

            expr.children = [massage2canonical_stage1(
                child, cases, levels+[1], mem, cases[best_case] if prev_case is not None else None) for child_index, child in enumerate(expr.children)]
            levels[-1] += 1
            expr = massage2canonical_stage1(expr, cases, levels, mem, cases[best_case] if prev_case is not None else None)
            mem[prev_expr] = expr
        else:
            break

    fix_structure(expr)

    if return_matches:
        return expr, had_matches
    else:
        return expr


def is_canonical(expr):
    """
    True if the expression is on
    the form AdX
    """
    if isinstance(expr, MatMulExpr):
        left, right = expr.children[0], expr.children[1]
        return isinstance(right, DifferentialExpr)
    else:
        return False
