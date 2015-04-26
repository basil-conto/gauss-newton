#!/usr/bin/env python

"""Provides solve(), an implementation of the Gauss-Newton algorithm.

This file also contains a sample program in main(), which runs solve() with a
couple of problems from the datasets module.
"""

import numpy as np

from datasets import Misra1a, Thurber

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2015.04.26"
__license__ = "BSD3"

def solve(sys, x0, tol = 1e-10, maxits = 500):
    """Gauss-Newton algorithm for solving nonlinear least squares problems.

    Parameters
    ----------
    sys : classobj
        Class providing model() and jacobian() functions. The former should
        evaluate an overdetermined nonlinear system given an n-tuple of
        arguments. The latter should evaluate the Jacobian matrix of said system
        given the same n-tuple of arguments.
    x0 : tuple
        Starting estimates for the system.
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
    dx = np.ones(len(x0))                                   # Correction vector

    i = 0
    while (i < maxits) and (dx[dx > tol].size > 0):
        b   = -sys.model(x0)                                # Residual vector
        dx  = np.dot(np.linalg.pinv(sys.jacobian(x0)), b)   # dx = pinv(J) . b
        x0 += dx                                            # x1 = x0 + dx
        i  += 1

    return x0, i

def main():
    """Solve the Misra1a and Thurber problems from the datasets module."""

    def output(sys, x0):
        sol, its = solve(sys, x0)
        cv = sys.cert_vals
        print("{}:".format(sys.__name__))
        print("  No. of iterations       : {}".format(its))
        print("  Calculated values       : {}".format(sol))
        print("  Certified  values       : {}".format(cv))
        print("  Diff. from cert. values : {}".format(np.abs(sol - cv)))

    # Nearby reference starting values
    output(Misra1a, (250, 5e-4))
    output(Thurber, (1300, 1500, 500, 74, 1, 0.4, 0.05))

if __name__ == "__main__":
    main()
