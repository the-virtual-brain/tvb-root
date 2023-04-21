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
Populate Surface fields after 1.3.1, in version 1.4.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER, dao, transactional
from tvb.core.utils import parse_json_parameters
from tvb.datatypes.region_mapping import RegionMapping
from tvb.storage.storage_interface import StorageInterface

LOGGER = get_logger(__name__)


def update():
    """
    Try to port Projection Matrices and Simulation Configurations from old to new form.
    """
    LOGGER.info("Start 7350 code update ...")

    _adapt_epileptor_simulations()

    _adapt_simulation_monitor_params()

    _transfer_projection_matrices()


@transactional
def _adapt_epileptor_simulations():
    """
    Previous Simulations on EpileptorWithPermitivity model, should be converted to use the Epileptor model.
    As the parameters from the two models are having different ranges and defaults, we do not translate parameters,
    we only set the Epileptor as model instead of EpileptorPermittivityCoupling, and leave the model params to defaults.
    """
    session = SA_SESSIONMAKER()
    epileptor_old = "EpileptorPermittivityCoupling"
    epileptor_new = "Epileptor"
    param_model = "model"

    try:
        all_ep_ops = session.query(model.Operation).filter(
            model.Operation.parameters.ilike('%"' + epileptor_old + '"%')).all()
        storage_interface = StorageInterface()
        all_bursts = dict()

        for ep_op in all_ep_ops:
            try:
                op_params = parse_json_parameters(ep_op.parameters)
                if op_params[param_model] != epileptor_old:
                    LOGGER.debug("Skipping op " + str(op_params[param_model]) + " -- " + str(ep_op))
                    continue

                LOGGER.debug("Updating " + str(op_params))
                op_params[param_model] = epileptor_new
                ep_op.parameters = json.dumps(op_params, cls=MapAsJson.MapAsJsonEncoder)
                LOGGER.debug("New params:" + ep_op.parameters)
                storage_interface.write_operation_metadata(ep_op)

                burst = dao.get_burst_for_operation_id(ep_op.id)
                if burst is not None:
                    LOGGER.debug("Updating burst:" + str(burst))
                    burst.prepare_after_load()
                    burst.simulator_configuration[param_model] = {'value': epileptor_new}
                    burst._simulator_configuration = json.dumps(burst.simulator_configuration,
                                                                cls=MapAsJson.MapAsJsonEncoder)
                    if burst.id not in all_bursts:
                        all_bursts[burst.id] = burst

            except Exception:
                LOGGER.exception("Could not process " + str(ep_op))

        session.add_all(all_ep_ops)
        session.add_all(list(all_bursts.values()))
        session.commit()

    except Exception:
        LOGGER.exception("Could not update Simulation Epileptor Params")
    finally:
        session.close()



@transactional
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
        all_bursts = dict()

        for eeg_op in all_eeg_ops:
            try:
                op_params = parse_json_parameters(eeg_op.parameters)
                LOGGER.debug("Updating " + str(op_params))
                old_projection_guid = op_params[param_eeg_proj_old]
                connectivity_guid = op_params[param_connectivity]

                rm = dao.get_generic_entity(RegionMapping, connectivity_guid, "_connectivity")[0]
                dt = dao.get_generic_entity(model.DataType, old_projection_guid, "gid")[0]

                if dt.type == 'ProjectionSurfaceEEG':
                    LOGGER.debug("Previous Prj is surface: " + old_projection_guid)
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

                burst = dao.get_burst_for_operation_id(eeg_op.id)
                if burst is not None:
                    LOGGER.debug("Updating burst:" + str(burst))
                    burst.prepare_after_load()
                    del burst.simulator_configuration[param_eeg_proj_old]
                    burst.simulator_configuration[param_eeg_proj_new] = {'value': str(new_projection_guid)}
                    burst.simulator_configuration[param_eeg_sensors] = {'value': str(sensors_guid)}
                    burst.simulator_configuration[param_eeg_rm] = {'value': str(rm.gid)}
                    burst._simulator_configuration = json.dumps(burst.simulator_configuration,
                                                                cls=MapAsJson.MapAsJsonEncoder)
                    if burst.id not in all_bursts:
                        all_bursts[burst.id] = burst

            except Exception:
                LOGGER.exception("Could not process " + str(eeg_op))

        session.add_all(all_eeg_ops)
        session.add_all(list(all_bursts.values()))
        session.commit()

    except Exception:
        LOGGER.exception("Could not update Simulation Monitor Params")
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

