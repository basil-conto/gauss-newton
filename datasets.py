#!/usr/bin/env python
"""Provides a class for nonlinear regression problems and instances thereof.

The data is taken from a NIST archive of Statistical Reference Datasets intended
for testing the robustness and reliability of nonlinear least squares problem
solvers. The archive can be found at:
http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

The Dataset class has been designed specifically for these nonlinear regression
datasets, so it may not be otherwise useful.

The Misra1a dataset, considered benign, is from a NIST study on dental research
in monomolecular adsorption. It involves an exponential model of 1 predictor
variable (x = pressure), 1 response variable (y = volume) and 2 parameters
(b1, b2). The dataset further includes 14 observed value pairs for x and y.
It can be found at: http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml

The Thurber dataset, considered challenging, is from a NIST study on
semiconductor electron mobility. It involves a rational model of 1 predictor
variable (x = log(density)), 1 response variable (y = electron mobility) and 7
parameters (b1, ..., b7). The dataset further includes 37 observed value pairs
for x and y.
It can be found at: http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
"""

from __future__ import division

import numpy as np
import sympy as sp

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2015.05.03"
__license__ = "BSD3"

class Dataset:

    """Representation of a NIST nonlinear regression dataset.

    The class attributes are the same as the constructor parameters.
    """

    def __init__(self, name, expr, symbols, xvals, yvals, cvals, starts):
        """Create a new Dataset.

        Parameters / Attributes
        -----------------------
        name : string
            Name of dataset, e.g. "Misra1a".
        expr : string
            Representation of dataset's model in a format understood by
            sympy.sympify().
        symbols : tuple or list
            SymPy Symbols found in `expr`. The first one should be the predictor
            variable and the rest are interpreted as model parameters.
        xvals : ndarray
            Observed or generated values for predictor variable.
        yvals : ndarray
            Observed or generated values for response variable.
        cvals : ndarray
            Certified values (i.e. reference solutions) for model parameters.
        starts : ndarray
            Nested set of initial guesses or starting estimates for the least
            squares solution of the system.
        """
        # Parameters become attributes
        self.name, self.starts  = name, starts
        self.expr, self.symbols = expr, symbols

        self.xvals, self.yvals, self.cvals = xvals, yvals, cvals

        # Predictor variable and parameters
        self._x, self._b = symbols[0], symbols[1:]

        # SymPy expression
        self._symexpr = sp.sympify(expr)
        # NumPy expression
        self._numexpr = sp.lambdify((self._x,) + self._b, self._symexpr, "numpy")
        # Partial derivatives
        self._pderivs = [self._symexpr.diff(b) for b in self._b]

    def __repr__(self):
        """Return Dataset description in the form <Dataset NAME at ADDRESS>."""
        return "<Dataset {} at {:x}>".format(self.name, id(self))

    def __str__(self):
        """Return name of Dataset, e.g. "Misra1a"."""
        return self.name

    def model(self, x = None, b = None):
        """Evaluate the model with the given predictor variable and parameters.

        Parameters
        ----------
        x : ndarray
            Values for the predictor variable. Defaults to the model's observed
            or generated values.
        b : tuple, list or ndarray
            Values for the model parameters. Defaults to their certified values.

        Return
        ------
        y : ndarray
            Corresponding values for the response variable.
        """
        if x is None: x = self.xvals
        if b is None: b = self.cvals
        return self._numexpr(x, *b)

    def system(self, b):
        """Evaluate f(x) - y with the given parameters.

        Parameters
        ----------
        b : tuple, list or ndarray
            Values for the model parameters.

        Return
        ------
        out : ndarray
            Evaluation of rearranged model.
        """
        x, y = self.xvals, self.yvals
        return self._numexpr(x, *b) - y

    def jacobian(self, b):
        """Evaluate the model's Jacobian matrix with the given parameters.

        Parameters
        ----------
        b : tuple, list or ndarray
            Values for the model parameters.

        Return
        ------
        out : ndarray
            Evaluation of the model's Jacobian matrix in column-major order wrt
            the model parameters.
        """
        # Substitute parameters in partial derivatives
        subs = [pd.subs(zip(self._b, b)) for pd in self._pderivs]
        # Evaluate substituted partial derivatives for all x-values
        vals = [sp.lambdify(self._x, sub, "numpy")(self.xvals) for sub in subs]
        # Arrange values in column-major order
        return np.column_stack(vals)

misra1a = Dataset(
       name = "Misra1a",
       expr = "b1 * (1 - exp(-b2 * x))",
    symbols = sp.symbols("x b1:3"),
      xvals = np.array(( 77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8,
                        378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0)),
      yvals = np.array((10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02,
                        44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78)),
      cvals = np.array((2.3894212918e+02, 5.5015643181e-04)),
     starts = np.array(((500, 0.0001), (250, 0.0005)))
)

thurber = Dataset(
       name = "Thurber",
       expr = "(b1 + (b2 * x) + (b3 * (x ** 2)) + (b4 * (x ** 3))) /" \
              "( 1 + (b5 * x) + (b6 * (x ** 2)) + (b7 * (x ** 3)))",
    symbols = sp.symbols("x b1:8"),
      xvals = np.array((
                -3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699,
                -2.633, -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212,
                -1.100, -1.046, -0.915, -0.714, -0.566, -0.545, -0.400, -0.309,
                -0.109, -0.103,  0.010,  0.119,  0.377,  0.790,  0.963,  1.006,
                 1.115,  1.572,  1.841,  2.047,  2.200
              )),
      yvals = np.array((
                  80.574,   84.248,   87.264,   87.195,   89.076,   89.608,
                  89.868,   90.101,   92.405,   95.854,  100.696,  101.060,
                 401.672,  390.724,  567.534,  635.316,  733.054,  759.087,
                 894.206,  990.785, 1090.109, 1080.914, 1122.643, 1178.351,
                1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509,
                1425.208, 1421.384, 1442.962, 1464.350, 1468.705, 1447.894,
                1457.628
              )),
      cvals = np.array((
                1.2881396800e+03, 1.4910792535e+03, 5.8323836877e+02,
                7.5416644291e+01, 9.6629502864e-01, 3.9797285797e-01,
                4.9727297349e-02
              )),
     starts = np.array(((1000, 1000, 400, 40, 0.7, 0.3, 0.03),
                        (1300, 1500, 500, 75, 1.0, 0.4, 0.05)))
)
