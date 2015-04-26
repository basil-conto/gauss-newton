#!/usr/bin/env python

"""Provides Misra1a and Thurber, classes representing nonlinear datasets.

The data is taken from a NIST archive of Statistical Reference Datasets intended
for testing the robustness and reliability of nonlinear least squares problem
solvers. The archive can be found at:
http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
"""

import numpy as np
import matplotlib.pyplot as plt

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2015.04.26"
__license__ = "BSD3"

class Misra1a:

    """Dataset from a NIST study on dental research in monomolecular adsorption.

    The Misra1a dataset, considered benign, involves an exponential model of
    1 predictor variable (x = pressure), 1 response variable (y = volume) and
    2 parameters (b1, b2). The dataset further includes 14 observed value pairs
    for x and y. It can be found at:
    http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml

    Attributes
    ----------
    x : ndarray
        14 observed x-values.
    y : ndarray
        14 observed y-values.
    cert_vals : ndarray
        Certified values (i.e. reference solutions) for x and y.
    """

    x = np.array(( 77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8,
                  378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0))

    y = np.array((10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02,
                  44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78))

    cert_vals = np.array((2.3894212918e+02, 5.5015643181e-04))

    @staticmethod
    def equation(x = None, b = tuple(None for i in xrange(2))):
        """Evaluate the Misra1a model with the given predictor and parameters.

        Defaults to the observed x-values and certified paramater values for any
        argument not specified.

        Parameters
        ----------
        x : ndarray
            Values for the predictor variable
        b : tuple
            2-tuple of parameters b1 and b2.

        Return
        ------
        y : ndarray
            Corresponding values for the response variable.
        """
        args = (x,) + b
        defs = (Misra1a.x,) + tuple(Misra1a.cert_vals)
        x, b1, b2 = [d if (a is None) else a for (a, d) in zip(args, defs)]
        return b1 * (1 - np.exp(-b2 * x))

    @staticmethod
    def model((b1, b2)):
        """Evaluate the nonlinear Misra1a model with the given parameters.

        The given parameters are used to evaluate a rearranged Misra1a model
        against observed values of x and y. The model is rearranged so that it
        can be equated to zero:
            (b1 * (1 - exp(-b2 * x))) - y

        Parameters
        ---------
        b : tuple
            2-tuple of parameters b1 and b2.

        Return
        ------
        out : ndarray
            Evaluation of rearranged Misra1a model.
        """
        M = Misra1a
        return (b1 * (1 - np.exp(-b2 * M.x))) - M.y

    @staticmethod
    def jacobian((b1, b2)):
        """Evaluate the Misra1a Jacobian matrix with the given parameters.

        The partial derivates corresponding to the Misra1a model are evaluated
        with respect to the given parameters b1 and b2. The resultant matrix is
        stored in column-major mode with respect to the parameters.

        Partial Derivates (wrt)
        -----------------------
        b1 : 1 - exp(-b2 * x)
        b2 : x * b1 * exp(-b2 * x)

        Parameters
        ---------
        b : tuple
            2-tuple of parameters b1 and b2.

        Return
        ------
        out : ndarray
            Evaluation of the Misra1a Jacobian matrix in column-major order wrt
            the parameters b1 and b2.
        """
        M = Misra1a
        e = np.exp(-b2 * M.x)
        return np.column_stack((1 - e, M.x * b1 * e))

class Thurber:

    """Dataset from a NIST study involving semiconductor electron mobility.

    The Thurber dataset, considered challenging, involves a rational model of
    1 predictor variable (x = log(density)), 1 response variable
    (y = electron mobility) and 7 parameters (b1, ..., b7). The dataset further
    includes 37 observed value pairs for x and y. It can be found at:
    http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml

    Attributes
    ----------
    x : ndarray
        37 observed x-values.
    y : ndarray
        37 observed y-values.
    cert_vals : ndarray
        Certified values (i.e. reference solutions) for x and y.
    """

    x = np.array((
        -3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699,
        -2.633, -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212,
        -1.100, -1.046, -0.915, -0.714, -0.566, -0.545, -0.400, -0.309,
        -0.109, -0.103,  0.010,  0.119,  0.377,  0.790,  0.963,  1.006,
         1.115,  1.572,  1.841,  2.047,  2.200
    ))

    y = np.array((
          80.574,   84.248,   87.264,   87.195,   89.076,   89.608,   89.868,
          90.101,   92.405,   95.854,  100.696,  101.060,  401.672,  390.724,
         567.534,  635.316,  733.054,  759.087,  894.206,  990.785, 1090.109,
        1080.914, 1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543,
        1353.863, 1414.509, 1425.208, 1421.384, 1442.962, 1464.350, 1468.705,
        1447.894, 1457.628
    ))

    cert_vals = np.array((
        1.2881396800e+03, 1.4910792535e+03, 5.8323836877e+02, 7.5416644291e+01,
        9.6629502864e-01, 3.9797285797e-01, 4.9727297349e-02
    ))

    @staticmethod
    def equation(x = None, b = tuple(None for i in xrange(7))):
        """Evaluate the Thurber model with the given predictor and parameters.

        Defaults to the observed x-values and certified paramater values for any
        argument not specified.

        Parameters
        ----------
        x : ndarray
            Values for the predictor variable
        b : tuple
            7-tuple of parameters (b1, ..., b7)

        Return
        ------
        y : ndarray
            Corresponding values for the response variable.
        """
        args = (x,) + b
        defs = (Thurber.x,) + tuple(Thurber.cert_vals)
        x, b1, b2, b3, b4, b5, b6, b7 = [d if (a is None) else a
                                         for (a, d) in zip(args, defs)]
        return ((b1 + (b2 * x) + (b3 * (x ** 2)) + (b4 * (x ** 3))) /
                ( 1 + (b5 * x) + (b6 * (x ** 2)) + (b7 * (x ** 3))))

    @staticmethod
    def model((b1, b2, b3, b4, b5, b6, b7)):
        """Evaluate the nonlinear Thurber model with the given parameters.

        The given parameters are used to evaluate a rearranged Thurber model
        against observed values of x and y. The model is rearranged so that it
        can be equated to zero:
            ((b1 + (b2 * x) + (b3 * (x ** 2)) + (b4 * (x ** 3))) /
             ( 1 + (b5 * x) + (b6 * (x ** 2)) + (b7 * (x ** 3)))) - y

        Parameters
        ---------
        b : tuple
            7-tuple of parameters (b1, ..., b7).

        Return
        ------
        out : ndarray
            Evaluation of rearranged Thurber model.
        """
        T = Thurber
        return ((b1 + (b2 * T.x) + (b3 * (T.x ** 2)) + (b4 * (T.x ** 3))) /
                ( 1 + (b5 * T.x) + (b6 * (T.x ** 2)) + (b7 * (T.x ** 3)))) - T.y

    @staticmethod
    def jacobian((b1, b2, b3, b4, b5, b6, b7)):
        """Evaluate the Thurber Jacobian matrix with the given parameters.

        The partial derivates corresponding to the Thurber model are evaluated
        with respect to the given parameters (b1, ..., b7). The resultant matrix
        is stored in column-major mode with respect to the parameters.

        Partial Derivates (wrt)
        -----------------------
        b1 :                1 / d
        b2 :  (b2 *  x)       / d
        b3 :  (b3 * (x ** 2)) / d
        b4 :  (b4 * (x ** 3)) / d
        b5 : -(b5 *  x)       * (n / (d ** 2))
        b6 : -(b6 * (x ** 2)) * (n / (d ** 2))
        b7 : -(b7 * (x ** 3)) * (n / (d ** 2))

        Where n and d are the model's numerator and denominator, respectively:
            n = b1 + (b2 * x) + (b3 * (x ** 2)) + (b4 * (x ** 3))
            d =  1 + (b5 * x) + (b6 * (x ** 2)) + (b7 * (x ** 3))

        Parameters
        ---------
        b : tuple
            7-tuple of parameters (b1, ..., b7).

        Return
        ------
        out : ndarray
            Evaluation of the Thurber Jacobian matrix in column-major order wrt
            the parameters (b1, ..., b7).
        """
        T = Thurber
        c2, c3, c4 = b2 * T.x, b3 * (T.x ** 2), b4 * (T.x ** 3)
        c5, c6, c7 = b5 * T.x, b6 * (T.x ** 2), b7 * (T.x ** 3)
        n  = b1 + c2 + c3 + c4
        d  =  1 + c5 + c6 + c7
        d2 = n / (d ** 2)
        return np.column_stack((1 / d, c2 / d, c3 / d, c4 / d,
                                -c5 * d2, -c6 * d2, -c7 * d2))
