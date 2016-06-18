"""
Test suite for experimental OpenCL components.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import unittest
import numpy
from tvb.tests.library.base_testcase import BaseTestCase

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

@unittest.skipIf(not PYOPENCL_AVAILABLE, 'PyOpenCL not available')
class TestModels(unittest.TestCase):
    def setUp(self):
        from tvb.simulator._opencl.util import create_cpu_context, context_and_queue
        self.context, self.queue = context_and_queue(create_cpu_context())
        self.n_nodes = 100




    def test_RWW_opencl(self):
        from tvb.simulator.models import ReducedWongWang
        from tvb.simulator._opencl.models import CLRWW

        self.validate(ReducedWongWang(),CLRWW(),1)

    #Passed
    def test_Kuramoto(self):
        from tvb.simulator.models.oscillator import  Kuramoto
        from tvb.simulator._opencl.cl_models import CL_Kuramoto
        self.validate(Kuramoto(), CL_Kuramoto(), CL_Kuramoto.n_states)

    #Passed
    def test_Generic2D(self):
        from tvb.simulator.models.oscillator import Generic2dOscillator
        from tvb.simulator._opencl.cl_models import CL_Generic2D
        self.validate(Generic2dOscillator(), CL_Generic2D(), CL_Generic2D.n_states)

    #Passed
    def test_Linear(self):
        from tvb.simulator.models.linear import Linear
        from tvb.simulator._opencl.cl_models import CL_Linear
        self.validate(Linear(), CL_Linear(), CL_Linear.n_states)

    #Passed
    def test_Epiletor(self):
        from tvb.simulator.models.epileptor import Epileptor
        from tvb.simulator._opencl.cl_models import CL_Epileptor
        self.validate(Epileptor(), CL_Epileptor(), CL_Epileptor.n_states)

    #Passed
    def test_Hopfiled(self):
        from tvb.simulator.models.hopfield import Hopfield
        from tvb.simulator._opencl.cl_models import CL_Hopfield
        self.validate(Hopfield(), CL_Hopfield(), CL_Hopfield.n_states)

    def test_Larter_Breakspear(self):
        from tvb.simulator.models.larter_breakspear import LarterBreakspear
        from tvb.simulator._opencl.cl_models import CL_Later_Breakspear
        self.validate(LarterBreakspear(), CL_Later_Breakspear(), CL_Later_Breakspear.n_states)


    def test_ReducedSet_FitzHughNagumo(self):
        from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
        from tvb.simulator._opencl.cl_models import CL_ReducedSetFitzHughNagumo
        self.validate(ReducedSetFitzHughNagumo(), CL_ReducedSetFitzHughNagumo(), CL_ReducedSetFitzHughNagumo.n_states)

    def test_ReducedSet_HindmarshRose(self):
        from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
        from tvb.simulator._opencl.cl_models import CL_ReducedSetHindmarshRose
        self.validate(ReducedSetHindmarshRose(), CL_ReducedSetHindmarshRose(), CL_ReducedSetHindmarshRose.n_states)

    def test_Wilson_Cowan(self):
        from tvb.simulator.models.wilson_cowan import WilsonCowan
        from tvb.simulator._opencl.cl_models import CL_WilsonCowan
        self.validate(WilsonCowan(), CL_WilsonCowan(), CL_WilsonCowan.n_states)

    def test_Zetterberg_Jasen(self):
        from tvb.simulator.models.jansen_rit import ZetterbergJansen
        from tvb.simulator._opencl.cl_models import CL_Zetterberg_Jasen
        self.validate(ZetterbergJansen(), CL_Zetterberg_Jasen(), CL_Zetterberg_Jasen.n_states)

    def validate(self, npm, clm, n_states):
        self.np = npm
        self.np.configure()
        self.cl = clm
        self.cl.configure()
        self.cl.configure_opencl(self.context, self.queue)
        self.state = numpy.random.rand(n_states, self.n_nodes, 1)
        self.coupling = numpy.random.rand(n_states, self.n_nodes, 1)

        np_dx = self.np.dfun(self.state, self.coupling)
        cl_dx = self.cl.dfun(self.state, self.coupling)
        numpy.testing.assert_allclose(cl_dx, np_dx, 1e-5, 1e-6)


# class TestCLModels(BaseTestCase):
#     def validate(self, model, clmodel ):
#         from tvb.simulator._opencl.util import create_cpu_context, context_and_queue
#         self.context, self.queue = context_and_queue(create_cpu_context())
#         self.clmodel = clmodel
#         self.model = model
#         self.n_nodes = 100
#         # self.state = numpy.random.rand(1, self.n_nodes, model.n_states)
#         # self.coupling = numpy.random.rand(1, self.n_nodes, model.n_states)
#         #
#         self.state = numpy.random.rand(1, self.n_nodes, 1)
#         self.coupling = numpy.random.rand(1, self.n_nodes, 1)
#
#         self.model.configure()
#         self.clmodel.configure_opencl(self.context, self.queue)
#
#         np_dx = self.model.dfun(self.state, self.coupling)
#         cl_dx = self.clmodel.dfun(self.state, self.coupling)
#
#         numpy.testing.assert_allclose(cl_dx, np_dx, 1e-5, 1e-6)
#     def test_RWW(self):
#         from tvb.simulator.models import ReducedWongWang
#         from tvb.simulator._opencl.models import CLRWW
#         self.validate(ReducedWongWang(),CLRWW())
#
#
#
# def suite():
#     """
#     Gather all the tests in a test suite.
#     """
#     test_suite = unittest.TestSuite()
#     test_suite.addTest(unittest.makeSuite(TestCLModels))
#     return test_suite
#
#
#
# if __name__ == "__main__":
#     #So you can run tests from this package individually.
#     TEST_RUNNER = unittest.TextTestRunner()
#     TEST_SUITE = suite()
#     TEST_RUNNER.run(TEST_SUITE)
