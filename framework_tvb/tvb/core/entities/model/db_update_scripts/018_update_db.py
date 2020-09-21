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
import json
import uuid
from migrate import create_column, drop_column, UniqueConstraint
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


def upgrade(migrate_engine):
    """
    """
    meta.bind = migrate_engine

    session = SA_SESSIONMAKER()

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

        # Dropping tables which will be repopulated from the H5 files
        session.execute(text("""DROP TABLE "BURST_CONFIGURATIONS";"""))
        session.execute(text("""DROP TABLE "MAPPED_COHERENCE_SPECTRUM_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_COMPLEX_COHERENCE_SPECTRUM_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_CONNECTIVITY_ANNOTATIONS_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_CONNECTIVITY_MEASURE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_CONNECTIVITY_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_CORRELATION_COEFFICIENTS_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_COVARIANCE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_CROSS_CORRELATION_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_FCD_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_FOURIER_SPECTRUM_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_INDEPENDENT_COMPONENTS_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_LOCAL_CONNECTIVITY_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_REGION_MAPPING_DATA";"""))
        session.execute(text("""DROP TABLE "ALGORITHMS";"""))
        session.execute(text("""DROP TABLE "ALGORITHM_CATEGORIES";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_REGION_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_EEG_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_MEG_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_SEEG_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_SURFACE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_TIME_SERIES_VOLUME_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SENSORS_DATA" """))
        session.execute(text("""DROP TABLE "MAPPED_STRUCTURAL_MRI_DATA" """))
        session.execute(text("""DROP TABLE "MAPPED_SURFACE_DATA" """))
        session.execute(text("""DROP TABLE "MAPPED_VOLUME_DATA" """))
        session.execute(text("""DROP TABLE "MAPPED_WAVELET_COEFFICIENTS_DATA";"""))
        session.execute(text("""DROP TABLE "DATA_TYPES";"""))

    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()

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
    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()

    UniqueConstraint("gid", table=users_table)

    # MIGRATING Operations #
    session = SA_SESSIONMAKER()
    try:
        burst_ref_metadata = session.execute(text("""SELECT id, parameters, meta_data FROM "OPERATIONS"
                WHERE meta_data like "%Burst_Reference%" """)).fetchall()

        for metadata in burst_ref_metadata:
            parameters_dict = eval(str(metadata[1]))
            metadata_dict = eval(str(metadata[2]))
            parameters_dict['parent_burst_id'] = metadata_dict['Burst_Reference']

            session.execute(text("""UPDATE "OPERATIONS" SET parameters = '""" + json.dumps(parameters_dict) +
                                 """' WHERE id = """ + str(metadata[0])))

        session.execute(text("""ALTER TABLE "OPERATIONS"
                                RENAME COLUMN parameters TO view_model_gid"""))
        session.commit()
    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()

    op_table = meta.tables['OPERATIONS']
    drop_column(OP_DELETED_COLUMN, op_table)


def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
