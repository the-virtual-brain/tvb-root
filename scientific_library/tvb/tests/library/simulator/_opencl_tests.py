"""
Test suite for experimental OpenCL components.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import unittest
import numpy

try:
    import pyopencl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False


@unittest.skipIf(not PYOPENCL_AVAILABLE, 'PyOpenCL not available')
class TestCLRWW(unittest.TestCase):

    def setUp(self):
        from tvb.simulator._opencl.util import create_cpu_context, context_and_queue
        self.context, self.queue = context_and_queue(create_cpu_context())
        self.n_nodes = 100
        self.state = numpy.random.rand(1, self.n_nodes, 1)
        self.coupling = numpy.random.rand(1, self.n_nodes, 1)

    def test_numpy_against_opencl(self):
        from tvb.simulator.models import ReducedWongWang
        from tvb.simulator._opencl.models import CLRWW
        np_rww = ReducedWongWang()
        np_rww.configure()
        cl_rww = CLRWW()
        cl_rww.configure()
        cl_rww.configure_opencl(self.context, self.queue)

        np_dx = np_rww.dfun(self.state, self.coupling)
        cl_dx = cl_rww.dfun(self.state, self.coupling)

        numpy.testing.assert_allclose(cl_dx, np_dx, 1e-5, 1e-6)