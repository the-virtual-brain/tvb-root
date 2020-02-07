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


class GenericAttributes(object):
    """
    Model class to hold generic attributes that we want to keep on all H5 file that correspond to a datatype.
    It is used by the H5File in order to populate the corresponding H5 attributes (meta-data).
    """
    invalid = False
    is_nan = False
    subject = ''
    state = ''
    type = ''
    user_tag_1 = ''
    user_tag_2 = ''
    user_tag_3 = ''
    user_tag_4 = ''
    user_tag_5 = ''
    visible = True
    create_date = None

    def fill_from(self, extra_attributes):
        # type: (GenericAttributes) -> None
        self.invalid = extra_attributes.invalid
        self.is_nan = extra_attributes.is_nan
        self.subject = extra_attributes.subject or self.subject
        self.state = extra_attributes.state or self.state
        self.type = extra_attributes.type or self.type
        self.user_tag_1 = extra_attributes.user_tag_1 or self.user_tag_1
        self.user_tag_2 = extra_attributes.user_tag_2 or self.user_tag_2
        self.user_tag_3 = extra_attributes.user_tag_3 or self.user_tag_3
        self.user_tag_4 = extra_attributes.user_tag_4 or self.user_tag_4
        self.user_tag_5 = extra_attributes.user_tag_5 or self.user_tag_5
        self.visible = extra_attributes.visible
