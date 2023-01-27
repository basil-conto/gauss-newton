#!/usr/bin/env python3

"""Provides solve(), an implementation of the Gauss-Newton algorithm.

This file also contains a sample program in main(), which runs solve() with a
couple of problems from the datasets module.
"""

import numpy as np

from datasets import misra1a, thurber

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2023.01.27"
__license__ = "BSD3"

def solve(sys, x0, tol=1e-10, maxits=256):
    """Gauss-Newton algorithm for solving nonlinear least squares problems.

    Parameters
    ----------
    sys : Dataset
        Class providing residuals() and jacobian() functions. The former should
        evaluate the residuals of a nonlinear system for a given set of
        parameters. The latter should evaluate the Jacobian matrix of said
        system for the same parameters.
    x0 : tuple, list or ndarray
        Initial guesses or starting estimates for the system.
    tol : float
        Tolerance threshold. The problem is considered solved when this value
        becomes smaller than the magnitude of the correction vector.
        Defaults to 1e-10.
    maxits : int
        Maximum number of iterations of the algorithm to perform.
        Defaults to 256.

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
        dx  = np.dot(np.linalg.pinv(sys.jacobian(xn)), -sys.residuals(xn))
        xn += dx            # x_{n + 1} = x_n + dx_n
        i  += 1

    return xn, i

def main():
    """Solve the Misra1a and Thurber problems from the datasets module."""

    # Inhibit wrapping of arrays in print
    np.set_printoptions(linewidth = 256)

    for ds in misra1a, thurber:
        for i, x0 in enumerate(ds.starts):
            sol, its = solve(ds, x0)
            cv = ds.cvals
            print(f"{ds}, start {i + 1}:")
            print(f"  Iterations : {its}")
            print(f"  Calculated : {sol}")
            print(f"  Certified  : {cv}")
            print(f"  Difference : {np.abs(sol - cv)}")

if __name__ == "__main__":
    main()
