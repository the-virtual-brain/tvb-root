# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
PCA analysis and visualisation demo.

``Run time``: approximately ? minutes (workstation circa 2010)

``Memory requirement``: ~ ?GB

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

from tvb.simulator.lab import *
from tvb.datatypes.time_series import TimeSeriesSurface
from tvb.simulator.plot import timeseries_interactive as timeseries_interactive
import tvb.analyzers.pca as pca


#Load the demo surface timeseries dataset 
try:
    data = numpy.load("demo_data_surface_8s_2048Hz.npy")
except IOError:
    LOG.error("Can't load demo data. Run demos/generate_surface_demo_data.py")
    raise

period = 0.00048828125  # s

#Initialse a default surface
default_cortex = surfaces.Cortex.from_file()

#Put the data into a TimeSeriesSurface datatype
tsr = TimeSeriesSurface(surface=default_cortex,
                        data=data,
                        sample_period=period)
tsr.configure()

#Create and run the analyser
pca_analyser = pca.PCA(time_series=tsr)
pca_data = pca_analyser.evaluate()

#Generate derived data, such as, component time series, etc.
pca_data.configure()

#Put the data into a TimeSeriesSurface datatype
component_tsr = TimeSeriesSurface(surface=default_cortex,
                                  data=pca_data.component_time_series,
                                  sample_period=period)
component_tsr.configure()

#Prutty pictures...
tsi = timeseries_interactive.TimeSeriesInteractive(time_series=component_tsr,
                                                   first_n=16)
tsi.configure()
tsi.show()

if IMPORTED_MAYAVI:
    surface_pattern(tsr.surface, pca_data.weights[:, 0, 0, 0])

