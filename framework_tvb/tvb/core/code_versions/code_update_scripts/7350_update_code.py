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

import json
from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_basic import MapAsJson
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER, dao
from tvb.core.utils import parse_json_parameters
from tvb.datatypes.region_mapping import RegionMapping


LOGGER = get_logger(__name__)



def update():
    """
    Try to port fProjection Matrices from old form into the new one
    """
    LOGGER.info("Start 7300 code update ...")

    _adapt_simulation_monitor_params()

    _transfer_projection_matrices()



def _adapt_simulation_monitor_params():
    """
    For previous simulation with EEG monitor, adjust the change of input parameters.
    """
    session = SA_SESSIONMAKER()

    param_connectivity = "connectivity"
    param_eeg_proj_old = "monitors_parameters_option_EEG_projection_matrix_data"
    param_eeg_proj_new = "monitors_parameters_option_EEG_projection"
    param_eeg_sensors = "monitors_parameters_option_EEG_sensors"
    param_eeg_rm = "monitors_parameters_option_EEG_region_mapping"

    try:
        all_eeg_ops = session.query(model.Operation).filter(
            model.Operation.parameters.ilike('%"' + param_eeg_proj_old + '"%')).all()

        for eeg_op in all_eeg_ops:
            try:
                op_params = parse_json_parameters(eeg_op.parameters)
                LOGGER.debug("Updating " + str(op_params))
                old_projection_guid = op_params[param_eeg_proj_old]
                connectivity_guid = op_params[param_connectivity]

                rm = dao.get_generic_entity(RegionMapping, connectivity_guid, "_connectivity")[0]
                dt = dao.get_generic_entity(model.DataType, old_projection_guid, "gid")[0]

                if dt.type == 'ProjectionSurfaceEEG':
                    LOGGER.debug("Previous Prj is surfac: " + old_projection_guid)
                    new_projection_guid = old_projection_guid
                else:
                    new_projection_guid = session.execute(text("""SELECT DT.gid
                            FROM "MAPPED_PROJECTION_MATRIX_DATA" PMO, "DATA_TYPES" DTO,
                                 "MAPPED_PROJECTION_MATRIX_DATA" PM, "DATA_TYPES" DT
                            WHERE DTO.id=PMO.id and DT.id=PM.id and PM._sensors=PMO._sensors and
                                  PM._sources='""" + rm._surface + """' and
                                  DTO.gid='""" + old_projection_guid + """';""")).fetchall()[0][0]
                    LOGGER.debug("New Prj is surface: " + str(new_projection_guid))

                sensors_guid = session.execute(text("""SELECT _sensors
                            FROM "MAPPED_PROJECTION_MATRIX_DATA"
                            WHERE id = '""" + str(dt.id) + """';""")).fetchall()[0][0]

                del op_params[param_eeg_proj_old]
                op_params[param_eeg_proj_new] = str(new_projection_guid)
                op_params[param_eeg_sensors] = str(sensors_guid)
                op_params[param_eeg_rm] = str(rm.gid)

                eeg_op.parameters = json.dumps(op_params, cls=MapAsJson.MapAsJsonEncoder)
                LOGGER.debug("New params:" + eeg_op.parameters)

            except Exception:
                LOGGER.exception("Could not process " + str(eeg_op))

        session.add_all(all_eeg_ops)
        session.commit()

        # TODO: update Burst configuration and operation.xml also.

    except Exception:
        LOGGER.exception("Could not update Simulation Params")
    finally:
        session.close()



def _transfer_projection_matrices():
    """
    Previous ProjectionRegionM/EEG objects should be Removed,
    and ProjectionSurfaceM/EEG should be transported into the new DB tables.
    """
    session = SA_SESSIONMAKER()
    LOGGER.info("Transferring Projections Surface ...")

    try:
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

        LOGGER.info("Removing Projections Region ...")

        session.execute(text("""DELETE from "DATA_TYPES"
                            WHERE type in ('ProjectionRegionEEG', 'ProjectionRegionMEG');"""))
        session.commit()

    except Exception:
        LOGGER.exception("Could not update Projection references")

    finally:
        session.close()

