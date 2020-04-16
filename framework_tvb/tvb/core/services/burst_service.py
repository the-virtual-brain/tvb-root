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
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from datetime import datetime
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.utils import format_bytes_human, format_timedelta

MAX_BURSTS_DISPLAYED = 50


class BurstService(object):

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()

    def mark_burst_finished(self, burst_entity, burst_status=None, error_message=None):
        """
        Mark Burst status field.
        Also compute 'weight' for current burst: no of operations inside, estimate time on disk...

        :param burst_entity: BurstConfiguration to be updated, at finish time.
        :param burst_status: BurstConfiguration status. By default BURST_FINISHED
        :param error_message: If given, set the status to error and perpetuate the message.
        """
        if burst_status is None:
            burst_status = BurstConfiguration.BURST_FINISHED
        if error_message is not None:
            burst_status = BurstConfiguration.BURST_ERROR

        try:
            # If there are any DataType Groups in current Burst, update their counter.
            burst_dt_groups = dao.get_generic_entity(DataTypeGroup, burst_entity.id, "fk_parent_burst")
            for dt_group in burst_dt_groups:
                dt_group.count_results = dao.count_datatypes_in_group(dt_group.id)
                dt_group.disk_size, dt_group.subject = dao.get_summary_for_group(dt_group.id)
                dao.store_entity(dt_group)

            # Update actual Burst entity fields
            burst_entity.datatypes_number = dao.count_datatypes_in_burst(burst_entity.id)

            burst_entity.status = burst_status
            burst_entity.error_message = error_message
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)
        except Exception:
            self.logger.exception("Could not correctly update Burst status and meta-data!")
            burst_entity.status = burst_status
            burst_entity.error_message = "Error when updating Burst Status"
            burst_entity.finish_time = datetime.now()
            dao.store_entity(burst_entity)

    def persist_operation_state(self, operation, operation_status, message=None):
        """
        Update Operation instance state. Store it in DB and on HDD/
        :param operation: Operation instance
        :param operation_status: new status
        :param message: message in case of error
        :return: operation instance changed
        """
        operation.mark_complete(operation_status, message)
        dao.store_entity(operation)
        operation = dao.get_operation_by_id(operation.id)
        self.file_helper.write_operation_metadata(operation)
        return operation

    def get_burst_for_operation_id(self, operation_id):
        return dao.get_burst_for_operation_id(operation_id)

    @staticmethod
    def rename_burst(burst_id, new_name):
        """
        Rename the burst given by burst_id, setting it's new name to
        burst_name.
        """
        burst = dao.get_burst_by_id(burst_id)
        burst.name = new_name
        dao.store_entity(burst)

    @staticmethod
    def get_available_bursts(project_id):
        """
        Return all the burst for the current project.
        """
        bursts = dao.get_bursts_for_project(project_id, page_size=MAX_BURSTS_DISPLAYED) or []
        # for burst in bursts:
        #     burst.prepare_after_load()
        return bursts

    @staticmethod
    def populate_burst_disk_usage(bursts):
        """
        Adds a disk_usage field to each burst object.
        The disk usage is computed as the sum of the datatypes generated by a burst
        """
        sizes = dao.compute_bursts_disk_size([b.id for b in bursts])
        for b in bursts:
            b.disk_size = format_bytes_human(sizes[b.id])

    def update_history_status(self, id_list):
        """
        For each burst_id received in the id_list read new status from DB and return a list [id, new_status] pair.
        """
        result = []
        for b_id in id_list:
            burst = dao.get_burst_by_id(b_id)
            # burst.prepare_after_load()
            if burst is not None:
                if burst.status == burst.BURST_RUNNING:
                    running_time = datetime.now() - burst.start_time
                else:
                    running_time = burst.finish_time - burst.start_time
                running_time = format_timedelta(running_time, most_significant2=False)

                if burst.status == burst.BURST_ERROR:
                    msg = 'Check Operations page for error Message'
                else:
                    msg = ''
                result.append([burst.id, burst.status, burst.is_group, msg, running_time])
            else:
                self.logger.debug("Could not find burst with id=" + str(b_id) + ". Might have been deleted by user!!")
        return result

    # TODO: We should implement these two methods
    def stop_burst(self, burst):
        raise NotImplementedError

    def cancel_or_remove_burst(self, burst_id):
        raise NotImplementedError

    @staticmethod
    def update_simulation_fields(burst_id, op_simulation_id, simulation_gid):
        burst = dao.get_burst_by_id(burst_id)
        burst.fk_simulation_id = op_simulation_id
        burst.simulator_gid = simulation_gid.hex
        dao.store_entity(burst)
