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
Change of DB structure from TVB version 1.4.1 to TVB 1.5

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import text
from migrate.changeset.constraint import ForeignKeyConstraint
from migrate.changeset.schema import create_column, drop_column
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER


meta = model.Base.metadata

LOGGER = get_logger(__name__)

DEL_COLUMNS = [Column('fk_algo_group', Integer),
               Column('identifier', String),
               Column('name', String)]

ADD_COLUMNS = [Column('fk_category', Integer),
               Column('module', String),
               Column('classname', String),
               Column('group_name', String),
               Column('group_description', String),
               Column('displayname', String),
               Column('subsection_name', String),
               Column('last_introspection_check', DateTime),
               Column('removed', Boolean, default=False)]



def upgrade(migrate_engine):
    """
    Alter existing table ALGORITHMS, by moving columns from the old ALGORITHM_GROUPS table.
    """
    meta.bind = migrate_engine
    table_algo = meta.tables["ALGORITHMS"]
    for col in ADD_COLUMNS:
        create_column(col, table_algo)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("ALTER TABLE \"MAPPED_SIMULATION_STATE\" "
                             "ADD COLUMN _current_state VARYING CHARACTER(255)"))
        session.commit()
    except Exception:
        session.close()
        session = SA_SESSIONMAKER()
        session.execute(text("ALTER TABLE \"MAPPED_SIMULATION_STATE\" "
                             "ADD COLUMN _current_state character varying;"))
        session.commit()
    finally:
        session.close()

    session = SA_SESSIONMAKER()
    try:
        session.execute(text(
            """UPDATE "ALGORITHMS" SET
            module = (select G.module FROM "ALGORITHM_GROUPS" G WHERE "ALGORITHMS".fk_algo_group=G.id),
            classname = (select G.classname FROM "ALGORITHM_GROUPS" G WHERE "ALGORITHMS".fk_algo_group=G.id),
            displayname = (select G.displayname FROM "ALGORITHM_GROUPS" G WHERE "ALGORITHMS".fk_algo_group=G.id),
            fk_category = (select G.fk_category FROM "ALGORITHM_GROUPS" G WHERE "ALGORITHMS".fk_algo_group=G.id);"""))
        session.commit()

        # Delete old columns, no longer needed
        for col in DEL_COLUMNS:
            drop_column(col, table_algo)

        # Create constraint only after rows are being populated
        table_algo = meta.tables["ALGORITHMS"]
        fk_constraint = ForeignKeyConstraint(["fk_category"], ["ALGORITHM_CATEGORIES.id"],
                                             ondelete="CASCADE", table=table_algo)
        fk_constraint.create()

        # Drop old table
        session = SA_SESSIONMAKER()
        session.execute(text("""DROP TABLE "ALGORITHM_GROUPS";"""))
        session.commit()

    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""ALTER TABLE "MAPPED_CONNECTIVITY_ANNOTATIONS"
                                RENAME TO "MAPPED_CONNECTIVITY_ANNOTATIONS_DATA"; """))
        session.execute(text("""ALTER TABLE "MAPPED_DATATYPE_MEASURE" RENAME TO "MAPPED_DATATYPE_MEASURE_DATA"; """))
        session.execute(text("""ALTER TABLE "MAPPED_SIMULATION_STATE" RENAME TO "MAPPED_SIMULATION_STATE_DATA"; """))
        session.execute(text("""ALTER TABLE "MAPPED_VALUE_WRAPPER" RENAME TO "MAPPED_VALUE_WRAPPER_DATA"; """))
        session.execute(text("""ALTER TABLE "MAPPED_PROJECTION_DATA" RENAME TO "MAPPED_PROJECTION_MATRIX_DATA"; """))
        session.commit()
    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()


def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
