# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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


ATT_NAME = "name"
ATT_TYPE = "type"
ATT_LABEL = "label"
ATT_DEFAULT = "default"
ATT_VALUE = "value"
ATT_FIELD = "field"
ATT_IDENTIFIER = "identifier"
ATT_ATTRIBUTES = "attributes"
ATT_DESCRIPTION = "description"
ATT_MINVALUE = "minValue"
ATT_MAXVALUE = "maxValue"
ATT_STEP = "step"
ATT_REQUIRED = "required"

ELEM_OPTIONS = "options"
ELEM_PORTLET = "portlet"
ELEM_INPUTS = "inputs"
ELEM_INPUT = "input"
ELEM_NAME = "name"
ELEM_LABEL = "label"
ELEM_DESCRIPTION = "description"
ELEM_TYPE = "type"
ELEM_CONDITIONS = "conditions"

TYPE_INT = "int"
TYPE_STR = "str"
TYPE_FLOAT = "float"
TYPE_BOOL = "bool"
TYPE_DICT = "dict"
TYPE_SELECT = "select"
TYPE_UPLOAD = "upload"
TYPE_MULTIPLE = "selectMultiple"
TYPE_ARRAY = "array"
TYPE_DYNAMIC = "dynamic"
TYPE_LIST = "list"


ALL_TYPES = (TYPE_STR, TYPE_FLOAT, TYPE_INT, TYPE_UPLOAD, TYPE_BOOL, TYPE_DICT,
             TYPE_ARRAY, TYPE_SELECT, TYPE_MULTIPLE, TYPE_DYNAMIC, TYPE_LIST)
