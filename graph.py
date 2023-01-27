#!/usr/bin/env python3

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
__version__ = "2023.01.27"
__license__ = "BSD3"

odir = "img/"
size = 10, 8
fmt = "co"

# Misra1a ----------------------------------------------------------------------
(ans0, _), (ans1, _) = [gn.solve(m, start) for start in m.starts]

# Observed range
x = m.xvals
plt.figure(figsize = size)
plt.plot(x, m.yvals, fmt,
         x, m.model(x, ans0),
         x, m.model(x, ans1),
         x, m.model())
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Observations", "Start 1", "Start 2", "Certified"], loc = "best")
plt.savefig(odir + "misra1a-obs.pdf", bbox_inches = "tight")
plt.close()

# Extended range
x = np.arange(-2048, 2048)
plt.figure(figsize = size)
plt.plot(x, m.model(x, ans0),
         x, m.model(x, ans1),
         x, m.model(x))
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Start 1", "Start 2", "Certified"], loc = "best")
plt.savefig(odir + "misra1a.pdf", bbox_inches = "tight")
plt.close()

# Thurber ----------------------------------------------------------------------
(ans0, _), (ans1, _) = [gn.solve(t, start) for start in t.starts]

# Observed range
x = t.xvals
plt.figure(figsize = size)
plt.plot(x, t.yvals, fmt,
         x, t.model(x, ans0),
         x, t.model(x, ans1),
         x, t.model())
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(ymin = 0)
plt.legend(["Observations", "Start 1", "Start 2", "Certified"], loc = "best")
plt.savefig(odir + "thurber-obs.pdf", bbox_inches = "tight")
plt.close()

# Extended range
x = np.arange(-25, 25)
plt.figure(figsize = size)
plt.plot(x, t.model(x, ans0),
         x, t.model(x, ans1),
         x, t.model(x))
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Start 1", "Start 2", "Certified"], loc = "best")
plt.savefig(odir + "thurber.pdf", bbox_inches = "tight")
plt.close()
