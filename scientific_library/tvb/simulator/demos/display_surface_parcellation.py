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

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

from tvb.simulator.lab import *
from tvb.simulator.region_boundaries import RegionBoundaries
from tvb.simulator.region_colours import RegionColours 


CORTEX = surfaces.Cortex.from_file()
CORTEX_BOUNDARIES = RegionBoundaries(CORTEX)

region_colours = RegionColours(CORTEX_BOUNDARIES.region_neighbours)
colouring = region_colours.back_track()

#Make the hemispheres symmetric
# TODO: should prob. et colouring for one hemisphere then just stack two copies...
number_of_regions = len(CORTEX_BOUNDARIES.region_neighbours)
for k in range(int(number_of_regions)):
    colouring[k + int(number_of_regions)] = colouring[k]


mapping_colours = list("rgbcmyRGBCMY")
colour_rgb = {"r": numpy.array([255,   0,   0], dtype=numpy.uint8),
              "g": numpy.array([  0, 255,   0], dtype=numpy.uint8),
              "b": numpy.array([  0,   0, 255], dtype=numpy.uint8),
              "c": numpy.array([  0, 255, 255], dtype=numpy.uint8),
              "m": numpy.array([255,   0, 255], dtype=numpy.uint8),
              "y": numpy.array([255, 255,   0], dtype=numpy.uint8),
              "R": numpy.array([128,   0,   0], dtype=numpy.uint8),
              "G": numpy.array([  0, 128,   0], dtype=numpy.uint8),
              "B": numpy.array([  0,   0, 128], dtype=numpy.uint8),
              "C": numpy.array([  0, 128, 128], dtype=numpy.uint8),
              "M": numpy.array([128,   0, 128], dtype=numpy.uint8),
              "Y": numpy.array([128, 128,   0], dtype=numpy.uint8)}


(surf_mesh, bpts) = surface_parcellation(CORTEX_BOUNDARIES, colouring, mapping_colours, colour_rgb, interaction=True)



##- EoF -##
