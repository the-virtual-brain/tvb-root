# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
#
#

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""


class GenericAttributes(object):
    """
    Model class to hold generic attributes that we want to keep on all H5 file that correspond to a datatype.
    It is used by the H5 File in order to populate the corresponding H5 attributes (meta-data).
    """
    invalid = False
    is_nan = False
    subject = ''
    state = ''
    user_tag_1 = ''
    user_tag_2 = ''
    user_tag_3 = ''
    user_tag_4 = ''
    user_tag_5 = ''
    operation_tag = ''
    parent_burst = None
    visible = True
    create_date = None

    def fill_from(self, extra_attributes):
        # type: (GenericAttributes) -> None
        self.invalid = extra_attributes.invalid
        self.is_nan = extra_attributes.is_nan
        self.subject = extra_attributes.subject or self.subject
        self.state = extra_attributes.state or self.state
        self.user_tag_1 = extra_attributes.user_tag_1 or self.user_tag_1
        self.user_tag_2 = extra_attributes.user_tag_2 or self.user_tag_2
        self.user_tag_3 = extra_attributes.user_tag_3 or self.user_tag_3
        self.user_tag_4 = extra_attributes.user_tag_4 or self.user_tag_4
        self.user_tag_5 = extra_attributes.user_tag_5 or self.user_tag_5
        self.operation_tag = extra_attributes.operation_tag or self.operation_tag
        self.parent_burst = extra_attributes.parent_burst or self.parent_burst
        self.visible = extra_attributes.visible
