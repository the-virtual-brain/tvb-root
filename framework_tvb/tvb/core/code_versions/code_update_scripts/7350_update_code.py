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
Populate Surface fields after 1.3.1, in version 1.4.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import SA_SESSIONMAKER


LOGGER = get_logger(__name__)



def update():
    """
    Try to port fProjection Matrices from old form into the new one
    """

    session = SA_SESSIONMAKER()

    try:

        LOGGER.info("Start 7300 code update ...")

        # Ony after SqlAlchemy finished initialization the new table MAPPED_PROJECTION_DATA exists
        session.execute(text("""INSERT into "MAPPED_PROJECTION_DATA" (id, _sources, _sensors, _projection_type)
                            SELECT PS.id, PM._sources, PM._sensors, 'projEEG'
                            FROM "MAPPED_PROJECTION_SURFACE_EEG_DATA" PS, "MAPPED_PROJECTION_MATRIX_DATA" PM
                            WHERE PM.id=PS.id;"""))

        session.execute(text("""INSERT into "MAPPED_PROJECTION_DATA" (id, _sources, _sensors, _projection_type)
                            SELECT PS.id, PM._sources, PM._sensors, 'projMEG'
                            FROM "MAPPED_PROJECTION_SURFACE_MEG_DATA" PS, "MAPPED_PROJECTION_MATRIX_DATA" PM
                            WHERE PM.id=PS.id;"""))

        session.execute(text("""DROP TABLE "MAPPED_PROJECTION_SURFACE_EEG_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_PROJECTION_SURFACE_MEG_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_PROJECTION_MATRIX_DATA";"""))

        session.execute(text("""DELETE from "DATA_TYPES"
                            WHERE type in ('ProjectionRegionEEG', 'ProjectionRegionMEG') ;"""))
        session.commit()

    except Exception:
        LOGGER.exception("Could update Projection references")

    finally:
        session.close()

