#!/usr/bin/env python

"""Provides solve(), an implementation of the Gauss-Newton algorithm.

This file also contains a sample program in main(), which runs solve() with a
couple of problems from the datasets module.
"""

import numpy as np

from datasets import misra1a, thurber

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2015.05.03"
__license__ = "BSD3"

def solve(sys, x0, tol = 1e-10, maxits = 500):
    """Gauss-Newton algorithm for solving nonlinear least squares problems.

    Parameters
    ----------
    sys : Dataset
        Class providing model() and jacobian() functions. The former should
        evaluate an overdetermined nonlinear system given an n-tuple of
        arguments. The latter should evaluate the Jacobian matrix of said system
        given the same n-tuple of arguments.
    x0 : tuple, list or ndarray
        Initial guesses or starting estimates for the system.
    tol : float
        Tolerance threshold. The problem is considered solved when this value
        becomes smaller than the magnitude of the correction vector.
    maxits : int
        Maximum number of iterations of the algorithm to perform.

    Return
    ------
    sol : ndarray
        Resultant values.
    its : int
        Number of iterations performed.

    Note
    ----
    Uses numpy.linalg.pinv() in place of similar functions from scipy, both
    because it was found to be faster and to eliminate the extra dependency.
    """
    dx = np.ones(len(x0))   # Correction vector
    xn = np.array(x0)       # Approximation of solution

    i = 0
    while (i < maxits) and (dx[dx > tol].size > 0):
        # correction = pinv(jacobian) . residual vector
        dx  = np.dot(np.linalg.pinv(sys.jacobian(xn)), -sys.system(xn))
        xn += dx            # x_{n + 1} = x_n + dx_n
        i  += 1

    return xn, i

def main():
    """Solve the Misra1a and Thurber problems from the datasets module."""

    # Inhibit wrapping of arrays in print
    np.set_printoptions(linewidth = 256)

    for sys in misra1a, thurber:
        for i, x0 in enumerate(sys.starts):
            sol, its = solve(sys, x0)
            cv = sys.cvals
            print("{}, start {}:".format(sys, i + 1))
            print("  Iterations : {}".format(its))
            print("  Calculated : {}".format(sol))
            print("  Certified  : {}".format(cv))
            print("  Difference : {}".format(np.abs(sol - cv)))

if __name__ == "__main__":
    main()
