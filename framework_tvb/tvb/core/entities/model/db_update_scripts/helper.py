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
# The Virtual Brain: a simulator of primate brain network dynamics.
# Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This modules holds helping function for DB update scripts

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import text
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import SA_SESSIONMAKER, dao
from tvb.core.utils import string2date

LOGGER = get_logger(__name__)


def change_algorithm(module, classname, new_module, new_class):
    """
    Change module and classname fields in ALGORITHM_GROUPS table.
    """
    session = SA_SESSIONMAKER()
    try:
        session.execute(text(
            """UPDATE "ALGORITHM_GROUPS"
               SET module = '""" + new_module + """', classname = '""" + new_class + """'
               WHERE module = '""" + module + """' AND classname = '""" + classname + """';"""))
        session.commit()
    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()


def get_burst_for_migration(burst_id, burst_match_dict, selected_db, date_format):
    """
    This method is supposed to only be used when migrating from version 4 to version 5.
    It finds a BurstConfig in the old format (when it did not inherit from HasTraitsIndex), deletes it
    and returns its parameters.
    """
    session = SA_SESSIONMAKER()
    burst_params = session.execute("""SELECT * FROM "BURST_CONFIGURATION" WHERE id = """ + burst_id).fetchone()

    if burst_params is None:
        return None, False

    if selected_db == 'sqlite':
        burst_params_dict = {'datatypes_number': burst_params[0], 'dynamic_ids': burst_params[1],
                             'range_1': burst_params[2], 'range_2': burst_params[3], 'fk_project': burst_params[5],
                             'name': burst_params[6], 'status': burst_params[7], 'error_message': burst_params[8],
                             'start_time': burst_params[9], 'finish_time': burst_params[10],
                             'fk_simulation': burst_params[12], 'fk_operation_group': burst_params[13],
                             'fk_metric_operation_group': burst_params[14]}
        burst_params_dict['start_time'] = string2date(burst_params_dict['start_time'], date_format=date_format)
        burst_params_dict['finish_time'] = string2date(burst_params_dict['finish_time'], date_format=date_format)
    else:
        burst_params_dict = {'fk_project': burst_params[1], 'name': burst_params[2], 'status': burst_params[3],
                             'error_message': burst_params[4], 'start_time': burst_params[5], 'finish_time':
                                 burst_params[6], 'datatypes_number': burst_params[7],
                             'dynamic_ids': burst_params[9],
                             'range_1': burst_params[10], 'range_2': burst_params[11], 'fk_simulation':
                                 burst_params[12], 'fk_operation_group': burst_params[13],
                             'fk_metric_operation_group': burst_params[14]}

    if burst_id not in burst_match_dict:
        burst_config = BurstConfiguration(burst_params_dict['fk_project'])
        burst_config.datatypes_number = burst_params_dict['datatypes_number']
        burst_config.dynamic_ids = burst_params_dict['dynamic_ids']
        burst_config.error_message = burst_params_dict['error_message']
        burst_config.finish_time = burst_params_dict['finish_time']
        burst_config.fk_metric_operation_group = burst_params_dict['fk_metric_operation_group']
        burst_config.fk_operation_group = burst_params_dict['fk_operation_group']
        burst_config.fk_project = burst_params_dict['fk_project']
        burst_config.fk_simulation = burst_params_dict['fk_simulation']
        burst_config.name = burst_params_dict['name']
        burst_config.range1 = burst_params_dict['range_1']
        burst_config.range2 = burst_params_dict['range_2']
        burst_config.start_time = burst_params_dict['start_time']
        burst_config.status = burst_params_dict['status']
        new_burst = True
    else:
        burst_config = dao.get_burst_by_id(burst_match_dict[burst_id])
        new_burst = False

    return burst_config, new_burst


def delete_old_burst_table_after_migration():
    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""DROP TABLE "BURST_CONFIGURATION"; """))
        session.commit()
    except Exception as excep:
        session.close()
        session = SA_SESSIONMAKER()
        LOGGER.exception(excep)
        try:
            session.execute(text("""DROP TABLE if exists "BURST_CONFIGURATION" cascade; """))
        except Exception as excep:
            LOGGER.exception(excep)
    finally:
        session.close()
