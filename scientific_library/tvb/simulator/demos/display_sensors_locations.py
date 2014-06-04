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

1) Plot MEG sensor locations and the region centres given by the
connectivity matrix.  

2) Plot EEG sensor locations on top of a surface representing the skin-air
boundary

NOTE: In general one assumes that coordinate systems are aligned, however ...
        
.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>

"""

from tvb.simulator.lab import *

##----------------------------------------------------------------------------##
##-                      Load datatypes                                      -##
##----------------------------------------------------------------------------##

# Get 'default' MEG sensors
sens_meg = sensors.SensorsMEG(load_default=True)

# Get connectivity
white_matter = connectivity.Connectivity(load_default=True)
centres = white_matter.centres

# Get surface - SkinAir
skin = surfaces.SkinAir(load_default=True)
skin.configure()

# Get 'default' EEG sensors
sens_eeg = sensors.SensorsEEG(load_default=True)
sens_eeg.configure()

# Project eeg unit vector locations onto the surface space
sensor_locations_eeg = sens_eeg.sensors_to_surface(skin)


#-----------------------------------------------------------------------------##
##-               Plot pretty pictures of what we just did                   -##
##----------------------------------------------------------------------------##

try:
    from tvb.simulator.plot.tools import mlab
    
    fig_meg = mlab.figure(figure='MEG sensors', bgcolor=(0.0, 0.0, 0.0))
    
    region_centres = mlab.points3d(centres[:, 0], 
                                   centres[:, 1], 
                                   centres[:, 2],
                                   color=(0.9, 0.9, 0.9),
                                   scale_factor = 10.)
    
    meg_sensor_loc = mlab.points3d(sens_meg.locations[:, 0],
                                   sens_meg.locations[:, 1], 
                                   sens_meg.locations[:, 2], 
                                   color=(0, 0, 1), 
                                   opacity = 0.6,
                                   scale_factor = 10,
                                   mode='cube')
    
    plot_surface(skin)
    eeg_sensor_loc = mlab.points3d(sensor_locations_eeg[:, 0],
                                   sensor_locations_eeg[:, 1],
                                   sensor_locations_eeg[:, 2],
                                   color=(0, 0, 1),
                                   opacity = 0.7,
                                   scale_factor=5)
    # Plot them
    mlab.show(stop=True)

except ImportError:
    LOG.exception("Could not display!")
    pass

# EoF