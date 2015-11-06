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
Framework methods for the Volume datatypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
from tvb.basic.traits import exceptions
from tvb.datatypes import volumes_data


class VolumeFramework(volumes_data.VolumeData):
    """ This class exists to add framework methods to VolumeData. """
    __tablename__ = None


class ParcellationMaskFramework(volumes_data.ParcellationMaskData,
                                VolumeFramework):
    """ This class exists to add framework methods to ParcellationMaskData. """
    pass


class StructuralMRIFramework(volumes_data.StructuralMRIData,
                             VolumeFramework):
    """ This class exists to add framework methods to StructuralMRIData. """
    pass


def preprocess_space_parameters(x, y, z, max_x, max_y, max_z):
    """
    Covert ajax call parameters into numbers and validate them.

    :param x:  coordinate
    :param y:  coordinate
    :param z:  coordinate that will be reversed
    :param max_x: maximum x accepted value
    :param max_y: maximum y accepted value
    :param max_z: maximum z accepted value

    :return: (x, y, z) as integers, Z reversed
    """

    x, y, z = int(x), int(y), int(z)

    if not 0 <= x <= max_x or not 0 <= y <= max_y or not 0 <= z <= max_z:
        msg = "Coordinates out of boundaries: [x,y,z] = [{0}, {1}, {2}]".format(x, y, z)
        raise exceptions.ValidationException(msg)

    # Reverse Z
    z = max_z - z - 1

    return x, y, z
