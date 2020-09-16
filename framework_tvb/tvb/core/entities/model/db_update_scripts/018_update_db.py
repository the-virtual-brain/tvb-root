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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import uuid

from migrate import create_column, drop_column, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy import Column, String, Integer
from sqlalchemy.engine import reflection
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import SA_SESSIONMAKER
from sqlalchemy.sql import text
from tvb.core.neotraits.db import Base

meta = Base.metadata

LOGGER = get_logger(__name__)


BURST_COLUMNS = [Column('range1', String), Column('range2', String), Column('fk_simulation', Integer),
Column('fk_operation_group', Integer), Column('fk_metric_operation_group', Integer)]
BURST_DELETED_COLUMN = Column('workflows_number', Integer)

OP_DELETED_COLUMN = Column('meta_data', String)

USER_COLUMNS = [Column('gid', String), Column('display_name', String)]


def migrate_range_params(ranges):
    new_ranges = []
    for range in ranges:
        list_range = eval(range)

        # in the range param name all the characters between the first and last underscores (including them)
        # must be deleted and replaced with a dot
        param_name = list_range[0]
        first_us = param_name.index('_')
        last_us = param_name.rfind('_')
        string_to_be_replaced = param_name[first_us:last_us + 1]
        param_name = '"' + param_name.replace(string_to_be_replaced, '.') + '"'

        # in the old version the range was a list of all values that the param had, but in the new one we
        # need only the minimum, maximum and step value
        param_range = list_range[1]
        range_dict = dict()
        range_dict['"' + 'lo' + '"'] = param_range[0]
        range_dict['"' + 'hi' + '"'] = param_range[-1]
        range_dict['"' + 'step' + '"'] = param_range[1] - param_range[0]

        new_ranges.append([param_name, range_dict])
    return new_ranges


def upgrade(migrate_engine):
    """
    """
    meta.bind = migrate_engine
    session = SA_SESSIONMAKER()

    # Renaming tables which wouldn't be correctly renamed by the next renamings
    try:
        session.execute(text("""ALTER TABLE "BURST_CONFIGURATIONS"
                                RENAME TO "BurstConfiguration"; """))
        session.execute(text("""ALTER TABLE "MAPPED_STRUCTURAL_MRI_DATA"
                                RENAME TO "StructuralMRIIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_EEG_DATA"
                                RENAME TO "TimeSeriesEEGIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_MEG_DATA"
                                RENAME TO "TimeSeriesMEGIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_SEEG_DATA"
                                RENAME TO "TimeSeriesSEEGIndex"; """))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    session = SA_SESSIONMAKER()
    inspector = reflection.Inspector.from_engine(session.connection())

    try:
        # Dropping tables which don't exist in the new version

        session.execute(text("""DROP TABLE "MAPPED_ARRAY_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_LOOK_UP_TABLE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_DATATYPE_MEASURE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_VOLUME_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SIMULATION_STATE_DATA";"""))
        session.execute(text("""DROP TABLE "WORKFLOWS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_STEPS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_VIEW_STEPS";"""))

        # This for renames the other tables replacing the "MAPPED_ ... _DATA" name structure with the "Ïndex" suffix
        for table in inspector.get_table_names():
            new_table_name = table
            new_table_name = new_table_name.lower()
            if 'mapped' in new_table_name:
                new_table_name = list(new_table_name)
                for i in range(len(new_table_name)):
                    if new_table_name[i] == '_':
                        new_table_name[i + 1] = new_table_name[i + 1].upper()

                new_table_name = "".join(new_table_name)
                new_table_name = new_table_name.replace('_', '')
                new_table_name = new_table_name.replace('mapped', '')
                new_table_name = new_table_name.replace('Data', '')
                new_table_name = new_table_name + 'Index'
                session.execute(text("""ALTER TABLE "{}" RENAME TO "{}"; """.format(table, new_table_name)))

        # Dropping ALGORITHMS and ALGORITHM CATEGORIES as they will be automatically generated at launching
        session.execute(text("""DROP TABLE "ALGORITHMS";"""))
        session.execute(text("""DROP TABLE "ALGORITHM_CATEGORIES";"""))
        session.execute(text("""DROP TABLE "TimeSeriesIndex";"""))
        session.execute(text("""DROP TABLE "TimeSeriesRegionIndex";"""))
        session.execute(text("""DROP TABLE "ConnectivityIndex";"""))
        session.execute(text("""DROP TABLE "DATA_TYPES";"""))

        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    # Migrating BurstConfiguration

    burst_config_table = meta.tables['BurstConfiguration']
    for column in BURST_COLUMNS:
        create_column(column, burst_config_table)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""ALTER TABLE "BurstConfiguration"
                                RENAME COLUMN _dynamic_ids TO dynamic_ids"""))
        session.execute(text("""ALTER TABLE "BurstConfiguration"
                                RENAME COLUMN _simulator_configuration TO simulator_gid"""))

        session.execute(text(
            """UPDATE "BurstConfiguration" SET
            fk_simulation = (SELECT O.id FROM "OPERATIONS" O, "DataType" D
             WHERE O.id = D.fk_from_operation AND module = 'tvb.adapters.datatypes.db.time_series')"""))

        ranges = session.execute(text("""SELECT OG.id, OG.range1, OG.range2
                            FROM "OPERATION_GROUPS" OG""")).fetchall()
        ranges_1 = []
        ranges_2 = []

        for r in ranges:
            ranges_1.append(str(r[1]))
            ranges_2.append(str(r[2]))

        new_ranges_1 = migrate_range_params(ranges_1)
        new_ranges_2 = migrate_range_params(ranges_2)

        #TODO: Fix this query to update the range params, for some reason it doesn't work

        # for i in range(len(ranges_1)):
        #     session.execute(text(
        #         """UPDATE "BurstConfiguration" SET
        #         range1 = """ + str(new_ranges_1[i]) + """,
        #         range2 = """ + str(new_ranges_2[i]) + """
        #         WHERE fk_operation_group = """ + str(ranges[i][0])))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    # Drop old column
    drop_column(BURST_DELETED_COLUMN, burst_config_table)

    # Create constraints only after the rows are populated
    fk_burst_config_constraint_1 = ForeignKeyConstraint(
        ["fk_simulation"],
        ["OPERATIONS.id"],
        table=burst_config_table)
    fk_burst_config_constraint_2 = ForeignKeyConstraint(
        ["fk_operation_group"],
        ["OPERATION_GROUPS.id"],
        table=burst_config_table)
    fk_burst_config_constraint_3 = ForeignKeyConstraint(
        ["fk_metric_operation_group"],
        ["OPERATION_GROUPS.id"],
        table=burst_config_table)

    fk_burst_config_constraint_1.create()
    fk_burst_config_constraint_2.create()
    fk_burst_config_constraint_3.create()

    # MIGRATING USERS #
    users_table = meta.tables['USERS']
    for column in USER_COLUMNS:
        create_column(column, users_table)

    session = SA_SESSIONMAKER()
    try:
        user_ids = eval(str(session.execute(text("""SELECT U.id
                            FROM "USERS" U """)).fetchall()))

        for id in user_ids:
            session.execute(text("""UPDATE "USERS" SET display_name = username,
                gid ='""" + uuid.uuid4().hex + """' WHERE id = """ + str(id[0])))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    UniqueConstraint("gid", table=users_table)
    # MIGRATING Operations #

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""ALTER TABLE "OPERATIONS"
                                RENAME COLUMN parameters TO view_model_gid"""))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    op_table = meta.tables['OPERATIONS']
    drop_column(OP_DELETED_COLUMN, op_table)


def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
