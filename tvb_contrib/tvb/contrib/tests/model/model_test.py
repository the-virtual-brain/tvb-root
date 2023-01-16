# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import numpy
import pytest
from tvb.basic.logger.builder import get_logger
from tvb.contrib.simulator.models.brunel_wang import BrunelWang
from tvb.contrib.simulator.models.epileptor import HMJEpileptor
from tvb.contrib.simulator.models.generic_2d_oscillator import Generic2dOscillator
from tvb.contrib.simulator.models.hindmarsh_rose import HindmarshRose
from tvb.contrib.simulator.models.jansen_rit_david import JansenRitDavid
from tvb.contrib.simulator.models.larter import Larter
from tvb.contrib.simulator.models.larter_breakspear import LarterBreakspear
from tvb.contrib.simulator.models.liley_steynross import LileySteynRoss
from tvb.contrib.simulator.models.morris_lecar import MorrisLecar
from tvb.contrib.simulator.models.wong_wang import WongWang
from tvb.simulator.integrators import HeunDeterministic
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

LOG = get_logger(__name__)


@pytest.mark.skip(reason="Because it opens a window for manual inspection")
class TestContribModels(BaseTestCase):

    @staticmethod
    def _show_model_figure(model_name, model_class, dt, **kwargs):
        # Do some stuff that tests or makes use of this module...
        LOG.info("Testing {} model...".format(model_name))

        # Check that the docstring examples, if there are any, are accurate.
        import doctest
        doctest.testmod()

        # Initialize Models in their default state
        model = model_class(**kwargs)

        LOG.info("Model initialized in its default state without error...")
        LOG.info("Testing phase plane interactive...")

        # Check the Phase Plane
        integrator = HeunDeterministic(dt=dt)
        ppi_fig = PhasePlaneInteractive(model=model, integrator=integrator)
        ppi_fig.show()

    def test_brunel_wang_model(self):
        self._show_model_figure("Brunel Wang", BrunelWang, 2 ** -5)

    def test_hmj_epileptor_model(self):
        self._show_model_figure("HMJEpileptor", HMJEpileptor, 2 ** -5)

    def test_generic_2d_oscillator_model(self):
        self._show_model_figure("Generic 2D Oscillator", Generic2dOscillator, 0.9)

    def test_hindmarsh_rose_model(self):
        self._show_model_figure("Hindmarsh Rose", HindmarshRose, 0.9)

    def test_jansen_rit_david_model(self):
        self._show_model_figure("Jansen Rit David", JansenRitDavid, 2 ** -5)

    def test_larter_model(self):
        self._show_model_figure("Larter", Larter, 0.9)

    def test_larter_breakspear_model(self):
        self._show_model_figure("Larter Breakspear", LarterBreakspear, 0.9, QV_max=numpy.array([1.0]),
                                QZ_max=numpy.array([1.0]), C=numpy.array([0.00]), d_V=numpy.array([0.6]),
                                aee=numpy.array([0.5]), aie=numpy.array([0.5]), gNa=numpy.array([0.0]),
                                Iext=numpy.array([0.165]), VT=numpy.array([0.65]), ani=numpy.array([0.1]))

    def test_liley_steynross(self):
        self._show_model_figure("Liley Steynross", LileySteynRoss, 0.9)

    def test_morric_lecar(self):
        self._show_model_figure("Morris Lecar", MorrisLecar, 2 ** -5)

    def test_wong_wang_model(self):
        self._show_model_figure("Wong Wang", WongWang, 2 ** -5)
