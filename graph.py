#!/usr/bin/env python

"""Graphs solutions from gaussnewton against their certified values.

Generates a graph for each dataset, including certified values and solutions
arrived at from both sets of starting points.
"""

import numpy as np
import gaussnewton as gn
import matplotlib.pyplot as plt

from datasets import misra1a as m
from datasets import thurber as t

__author__  = "Basil L. Contovounesios"
__email__   = "contovob@tcd.ie"
__version__ = "2015.05.03"
__license__ = "BSD3"

def save(sys, x, title = None, fname = "plot.pdf", odir = ""):
    (ans0, _), (ans1, _) = [gn.solve(sys, start) for start in sys.starts]
    plt.figure()
    plt.plot(x, sys.model(x, ans0),
             x, sys.model(x, ans1),
             x, sys.model(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Start1", "Start2", "Certified"], loc = "best")
    plt.title(title)
    plt.savefig(odir + fname, bbox_inches = "tight")
    plt.close()

odir = "img/"

save(m, m.xvals, "Misra1a Solution - Observed Range", "misra1a-obs.pdf", odir)
save(m, np.arange(-2048, 2048), "Misra1a Solution", "misra1a.pdf", odir)

save(t, t.xvals, "Thurber Solution - Observed Range", "thurber-obs.pdf", odir)
save(t, np.arange(-25, 25), "Thurber Solution", "thurber.pdf", odir)
