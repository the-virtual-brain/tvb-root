# we need a global otherwise cython occasionally forgets to put a module init! A docstring will do it
"""
Accelerated dfuns
"""
import numpy as np
cimport numpy as np
cimport cython

# Local coupling is a 0.0 for region simulations. For surface ones it is a sparse matrix.
# This means that we have to declare it as object.
# Not ideal. Consider moving this local coupling multiplication to python

#@cython.boundscheck(False)
def epileptor_dfun(np.ndarray[double, ndim=3] y, # sv , n , m
         np.ndarray[double, ndim=3] coupling,  #ncvar, n, m
         object local_coupling,
         double a, double b, double c, double d, double aa, double r,
         double Kvf, double Kf, double tau, double Iext, double Iext2,
         double slope, double x0
         ):
    cdef int nodes, modes
    nodes = y.shape[1]
    modes = y.shape[2]

    cdef double y0, y1, y2, y3, y4, y5
    cdef double tmp, iext_inner, cpop_1, cpop_2

    cdef np.ndarray[double, ndim=3, mode="c"] ydot
    cdef np.ndarray[double, ndim=2] Iext_nd

    ydot = np.empty_like(y)

    Iext_nd = Iext + local_coupling * y[0, :, :]

    for ns in range(nodes):
        for m in range(modes):
            y0 = y[0, ns, m]
            y1 = y[1, ns, m]
            y2 = y[2, ns, m]
            y3 = y[3, ns, m]
            y4 = y[4, ns, m]
            y5 = y[5, ns, m]
            iext_inner = Iext_nd[ns, m]
            cpop_1 = coupling[0, ns, m]
            cpop_2 = coupling[1, ns, m]

            # population 1
            if y0 < 0.0:
                tmp = - a * y0**2 + b * y0
            else:
                tmp = slope - y3 + 0.6 * (y2 - 4.0)**2

            ydot[0, ns, m] = y1 - y2 + iext_inner + Kvf * cpop_1 + tmp * y0
            ydot[1, ns, m] = c - d * y0**2 - y1

            # energy
            if y2 < 0.0:
                tmp =  - 0.1 * y2**7
            else:
                tmp = 0

            ydot[2, ns, m] = r * ( 4 * (y0 - x0) - y2 + tmp )

            # population 2
            ydot[3, ns, m] = -y4 + y3 - y3**3 + Iext2 + 2 * y5 - 0.3 * (y2 - 3.5) + Kf * cpop_2
            if y3 < -0.25:
                tmp = 0
            else:
                tmp = aa * (y3 + 0.25)

            ydot[4, ns, m] = (-y4 + tmp) / tau

            # filter
            ydot[5, ns, m] = -0.01 * (y5 - 0.1 * y0)

    return ydot
