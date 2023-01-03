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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from datetime import datetime
import uuid


class DummyProject:

    def __init__(self, name, description, version, fk_admin):
        self.name = name
        self.description = description
        self.last_updated = datetime.now()
        self.gid = uuid.uuid4().hex
        self.version = version
        self.fk_admin = fk_admin

    def to_dict(self):
        return {"name": self.name, "description": self.description, "last_updated": self.last_updated,
                "gid": self.gid, "version": self.version}

    def from_dict(self, dict, user_id):
        self.name = dict["name"]
        self.description = dict["description"]
        self.last_updated = dict["last_updated"]
        self.gid = dict["gid"]
        self.version = dict["version"]
        self.fk_admin = user_id
