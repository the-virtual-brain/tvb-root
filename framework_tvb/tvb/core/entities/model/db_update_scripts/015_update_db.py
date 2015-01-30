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
Change of DB structure from TVB version 1.1.2 to 1.1.3

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER, dao


meta = model.Base.metadata

LOGGER = get_logger(__name__)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    try:
        meta.bind = migrate_engine
        session = SA_SESSIONMAKER()
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.region_mapping' WHERE "type" = 'RegionMapping' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.local_connectivity' WHERE "type" = 'LocalConnectivity' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.cortex' WHERE "type" = 'Cortex' """))
        session.commit()
        session.close()

    except Exception:
        LOGGER.exception("Cold not update datatypes")
        raise


def downgrade(migrate_engine):
    """
    Operations to reverse the above upgrade go here.
    """
    try:
        meta.bind = migrate_engine
        session = SA_SESSIONMAKER()
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'RegionMapping' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'LocalConnectivity' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'Cortex' """))
        session.commit()
        session.close()
    except Exception:
        LOGGER.exception("Cold not update datatypes")
        raise