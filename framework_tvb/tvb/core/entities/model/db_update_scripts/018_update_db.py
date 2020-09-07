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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Change of DB structure to TVB 2.0

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
from sqlalchemy import inspect
from sqlalchemy.engine import reflection
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import SA_SESSIONMAKER
from sqlalchemy.sql import text
from tvb.core.neotraits.db import Base


meta = Base.metadata

LOGGER = get_logger(__name__)


def upgrade(migrate_engine):
    """
    """
    meta.bind = migrate_engine

    session = SA_SESSIONMAKER()
    inspector = reflection.Inspector.from_engine(session.connection())

    try:
        for table in inspector.get_table_names():
            new_table_name = table
            new_table_name = new_table_name.lower()
            if 'mapped' in new_table_name:
                for i in range(len(new_table_name)):
                    if new_table_name[i] == '_':
                        new_table_name[i+1] = new_table_name[i+1].upper()

                new_table_name.replace('_', '')
                new_table_name.replace('MAPPED', '')
                new_table_name.replace('DATA', '')
                new_table_name.append('Index')

                session.execute(text("""ALTER TABLE "{}" RENAME TO "{}"; """.format(table, new_table_name)))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
