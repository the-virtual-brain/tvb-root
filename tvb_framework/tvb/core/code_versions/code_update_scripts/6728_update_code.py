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
Populate Surface fields after 1.2.2, in version 1.2.3.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import dao
from tvb.core.entities import model
from tvb.datatypes.surfaces import Surface


LOGGER = get_logger(__name__)

def update():
    """
    Update Surface metadata
    """

    try:
        all_surfaces = dao.get_generic_entity(model.DataType, "tvb.datatypes.surfaces", "module")
        for srf in all_surfaces:
            surface = dao.get_datatype_by_gid(srf.gid)
            if isinstance(surface, Surface):
                surface.configure()
                dao.store_entity(surface)
                surface.persist_full_metadata()
    except Exception:
        LOGGER.exception("Could update Surface entities!")
