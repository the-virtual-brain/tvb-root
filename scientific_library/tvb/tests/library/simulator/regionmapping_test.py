"""
Tests full region mapping when both surface and sub-cortical regions are present.

.. moduleauthor:: Borana Dollomaja <borana.dollomaja@univ-amu.fr>
"""
import pytest
import numpy
import time
# import TVB modules
from tvb.simulator.lab import *
from tvb.datatypes.region_mapping import RegionMapping
from tvb.tests.library.base_testcase import BaseTestCase


class TestRegionMapping(BaseTestCase):

    def setUp(self, monitors=(monitors.Raw(),)):
        """
        Initialize the structural information, coupling function, integrator, 
        monitors, surface and stimulation.
        """
        # Connectome
        con = connectivity.Connectivity.from_file("connectivity_192.zip")
        con.configure()

        # Surface and local connectivity kernel
        surf = cortex.Cortex.from_file() #Initialise a surface
        surf.local_connectivity = local_connectivity.LocalConnectivity.from_file()
        surf.configure()

        # Model
        oscilator = models.Generic2dOscillator()

        # monitors[0].spatial_mask = RegionMapping.full_region_mapping(surf, con)

        self.sim = simulator.Simulator(
            conduction_speed=1.0,
            coupling= coupling.Difference(a=numpy.array([0.01])),
            surface=surf,
            integrator=integrators.Identity(dt=1.0),
            simulation_length=10.0,
            connectivity=con,
            model=oscilator,
            monitors=monitors
        )
        self.sim.configure()

    def test_spatial_average_monitor(self):
        self.setUp(monitors=(monitors.SpatialAverage(period=1.),))
        result = self.sim.run(simulation_length = 10)
        assert result is not None