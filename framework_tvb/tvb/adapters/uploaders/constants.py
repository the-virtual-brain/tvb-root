# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

### Connectivity related constants
KEY_CONNECTIVITY_UID = 'CONNECTIVITY_UID'


### Surface /Cortex related constants
ROLE_REGION_MAP = 'region_map'
ROLE_LOCAL_CON = 'local_con'
ROLE_VERTICES = 'ROLE_VERTICES'
ROLE_NORMALS = 'ROLE_NORMALS'
ROLE_TRIANGLES = 'ROLE_TRIANGLES'
DATA_NAME_LOCAL_CONN = "LocalCoupling"
DATA_NAME_PROJECTION = "ProjectionMatrix"

PARAM_ZERO_BASED = 'ZERO_BASED'
MAPPINGS_DICT = 'SURFACE_MAPPINGS'
SURFACE_TYPE = 'SURFACE_TYPE'
KEY_SURFACE_UID = 'SURFACE_UID'


### These constants are surface type names as shown in the ui select boxes
OPTION_SURFACE_FACE = "Face"
OPTION_SURFACE_SKINAIR = "SkinAir"
OPTION_SURFACE_CORTEX = "Cortex"


### TimeSeries related constants.
LABEL_X = "label_x"
LABEL_Y = "label_y"
SHAPE = "shape"
MAX_CHUNKS_LENGTH = "max_chunks_length"
CHANNEL_MAX_VALUES = "channel_max_values"
CHANNEL_MIN_VALUES = "channel_min_values"
GID = "gid"
SUB_ARRAY = "sub_array"
DISPLAY_NAME = "display_name"


### Generic CFF constants.
KEY_ROLE = 'ROLE'
KEY_UID = 'UID'
SURFACE_ACTIVITY_TYPE = "surface_activity_type"
MIN_ACTIVITY = "min_activity_value"
MAX_ACTIVITY = "max_activity_value"
SURFACE_CLASS = "surface_class"
CLASS_SURFACE = "Surface"
CLASS_CORTEX = "Cortex"
CLASS_CORTEX_ACTIVITY = "CortexActivity"

