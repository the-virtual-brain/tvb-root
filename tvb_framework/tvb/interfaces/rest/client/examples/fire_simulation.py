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
Example of launching a simulation and an analyzer from within the REST client API.
"""

from tvb.adapters.analyzers.fourier_adapter import FFTAdapterModel, FourierAdapter
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesH5
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.datatypes.spectral import WindowingFunctionsEnum
from tvb.interfaces.rest.client.examples.utils import monitor_operation, compute_rest_url
from tvb.interfaces.rest.client.tvb_client import TVBClient

logger = get_logger(__name__)


def fire_simulation_example(tvb_client_instance):
    logger.info("Requesting projects for logged user")
    projects_of_user = tvb_client_instance.get_project_list()
    assert len(projects_of_user) > 0
    logger.info("TVB has {} projects for this user".format(len(projects_of_user)))

    project_gid = projects_of_user[0].gid
    logger.info("Requesting datatypes from project {}...".format(project_gid))
    data_in_project = tvb_client_instance.get_data_in_project(project_gid)
    logger.info("We have {} datatypes".format(len(data_in_project)))

    logger.info("Requesting operations from project {}...".format(project_gid))
    ops_in_project, _ = tvb_client_instance.get_operations_in_project(project_gid, 1)
    logger.info("Displayname of the first operation is: {}".format(ops_in_project[0].displayname))

    connectivity_gid = None
    datatypes_type = []
    for datatype in data_in_project:
        datatypes_type.append(datatype.type)
        if datatype.type == ConnectivityIndex().display_type:
            connectivity_gid = datatype.gid
    logger.info("The datatypes in project are: {}".format(datatypes_type))

    if connectivity_gid:
        logger.info("Preparing the simulator...")
        simulator = SimulatorAdapterModel()
        simulator.connectivity = connectivity_gid
        simulator.simulation_length = 100

        logger.info("Starting the simulation...")
        operation_gid = tvb_client_instance.fire_simulation(project_gid, simulator)

        logger.info("Monitoring the simulation operation...")
        monitor_operation(tvb_client_instance, operation_gid)

        logger.info("Requesting the results of the simulation...")
        simulation_results = tvb_client_instance.get_operation_results(operation_gid)
        datatype_names = []
        for datatype in simulation_results:
            datatype_names.append(datatype.name)
        logger.info("The resulted datatype are: {}".format(datatype_names))

        time_series_gid = simulation_results[1].gid
        logger.info("Download the time series file...")
        time_series_path = tvb_client_instance.retrieve_datatype(time_series_gid,
                                                                 tvb_client_instance.temp_folder)

        logger.info("The time series file location is: {}".format(time_series_path))

        logger.info("Requesting algorithms to run on time series...")
        algos = tvb_client_instance.get_operations_for_datatype(time_series_gid)
        algo_names = [algo.displayname for algo in algos]
        logger.info("Possible algorithms are {}".format(algo_names))

        logger.info("Launch Fourier Analyzer...")
        fourier_model = FFTAdapterModel()
        fourier_model.time_series = time_series_gid
        fourier_model.window_function = WindowingFunctionsEnum.HAMMING

        operation_gid = tvb_client_instance.launch_operation(project_gid, FourierAdapter, fourier_model)
        logger.info("Fourier Analyzer operation has launched with gid {}".format(operation_gid))

        data_in_project = tvb_client_instance.get_data_in_project(project_gid)
        logger.info("We have {} datatypes".format(len(data_in_project)))

        for datatype in data_in_project:
            if datatype.type == 'FourierSpectrum':
                ggid = datatype.gid

                extra_info = tvb_client_instance.get_extra_info(ggid)
                logger.info("The extra information for Fourier {}".format(extra_info))

                break

        logger.info("Download the connectivity file...")
        connectivity_path = tvb_client_instance.retrieve_datatype(connectivity_gid,
                                                                  tvb_client_instance.temp_folder)

        logger.info("The connectivity file location is: {}".format(connectivity_path))

        logger.info("Loading an entire Connectivity datatype in memory...")
        connectivity = tvb_client_instance.load_datatype_from_file(connectivity_path)
        logger.info("Info on current Connectivity: {}".format(connectivity.summary_info()))

        logger.info("Loading a chuck from the time series H5 file, as this can be very large...")
        with TimeSeriesH5(time_series_path) as time_series_h5:
            data_shape = time_series_h5.read_data_shape()
            chunk = time_series_h5.read_data_slice(
                tuple([slice(20), slice(data_shape[1]), slice(data_shape[2]), slice(data_shape[3])]))

        assert chunk.shape[0] == 20
        assert chunk.shape[1] == data_shape[1]
        assert chunk.shape[2] == data_shape[2]
        assert chunk.shape[3] == data_shape[3]

        return project_gid, time_series_gid


def quick_launch_an_operation(tvb_client_instance, project_gid, datatype_gid):
    """
    This is intended to simulate the following behavior:
        - in GUI the user has the possibility to view all algorithms that can run over a certain datatype
        - he chooses a datatype and all these algorithms are displayed
        - then, he selects an algorithm
    The data that should be sent to the server:
        - the selected datatype_gid
        - the index of the selected algorithm
    """
    algo_dto_list = tvb_client_instance.get_operations_for_datatype(datatype_gid)
    algo_dto_index = 0

    # Supposing that algo_dt_index and time_series_gid are sent from the client-side
    operation_gid = tvb_client.quick_launch_operation(project_gid, algo_dto_list[algo_dto_index], datatype_gid)
    monitor_operation(tvb_client, operation_gid)


if __name__ == '__main__':
    logger.info("Preparing client...")
    tvb_client = TVBClient(compute_rest_url())

    logger.info("Attempt to login")
    tvb_client.browser_login()
    project_gid, time_series_gid = fire_simulation_example(tvb_client)
    quick_launch_an_operation(tvb_client, project_gid, time_series_gid)
