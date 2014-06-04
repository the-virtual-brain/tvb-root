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

The Connectivity datatype. This brings together the scientific and framework 
methods that are associated with the connectivity data.

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

import numpy
from tvb.datatypes import connectivity_scientific
from tvb.datatypes import connectivity_framework
from tvb.basic.readers import ZipReader, H5Reader, try_get_absolute_path



class Connectivity(connectivity_scientific.ConnectivityScientific, connectivity_framework.ConnectivityFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the Connectivity dataType.
    
    ::
        
                          ConnectivityData
                                 |
                                / \\
          ConnectivityFramework     ConnectivityScientific
                                \ /
                                 |
                            Connectivity
        
    
    """

    @staticmethod
    def from_file(source_file="connectivity_74.zip", instance=None):

        if instance is None:
            result = Connectivity()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.connectivity", source_file)

        if source_file.endswith(".h5"):

            reader = H5Reader(source_full_path)

            result.weights = reader.read_field("weights")
            result.centres = reader.read_field("centres")
            result.region_labels = reader.read_field("region_labels")
            result.orientations = reader.read_field("orientations")
            result.cortical = reader.read_field("cortical")
            result.hemispheres = reader.read_field("hemispheres")
            result.areas = reader.read_field("areas")
            result.tract_lengths = reader.read_field("tract_lengths")

        else:
            reader = ZipReader(source_full_path)

            result.weights = reader.read_array_from_file("weights.txt")
            result.centres = reader.read_array_from_file("centres.txt", use_cols=(1, 2, 3))
            result.region_labels = reader.read_array_from_file("centres.txt", dtype="string", use_cols=(0,))
            result.orientations = reader.read_array_from_file("average_orientations.txt")
            result.cortical = reader.read_array_from_file("cortical.txt", dtype=numpy.bool)
            result.hemispheres = reader.read_array_from_file("hemispheres.txt", dtype=numpy.bool)
            result.areas = reader.read_array_from_file("areas.txt")
            result.tract_lengths = reader.read_array_from_file("tract_lengths.txt")

        return result

    @classmethod
    def default(cls):
        return cls.from_file()

