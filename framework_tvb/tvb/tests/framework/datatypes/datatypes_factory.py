# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
This module contains methods for creating persisted data-types for tests.

.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import numpy
import time
from tvb.config.init.introspector_registry import IntrospectionRegistry
from tvb.core.entities.model.model_operation import *
from tvb.core.entities.model.model_datatype import *
from tvb.core.entities.model.datatypes.mapped_value import DatatypeMeasureIndex, ValueWrapperIndex
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.equations import PulseTrain
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.time_series import TimeSeries, TimeSeriesEEG, TimeSeriesRegion
from tvb.datatypes.graph import Covariance, ConnectivityMeasure
from tvb.datatypes.spectral import CoherenceSpectrum
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.mode_decompositions import IndependentComponents


class DatatypesFactory(object):
    """
    This class provides a set of methods that helps user to create
    different data types for testing.
    These data types will be automatically stored in DB and file system if needed.
    """
    USER_FULL_NAME = "Datatype Factory User"
    DATATYPE_STATE = "RAW_DATA"
    DATATYPE_DATA = ["test", "for", "datatypes", "factory"]

    DATATYPE_MEASURE_METRIC = {'v': 3}
    RANGE_1 = ["row1", [1, 2, 3]]
    RANGE_2 = ["row2", [0.1, 0.3, 0.5]]

    user = None
    project = None
    operation = None

    def __init__(self):
        micro_postfix = "_%d" % int(time.time() * 1000000)

        # Here create all structures needed later for data types creation
        self.files_helper = FilesHelper()

        # First create user 
        user = User("datatype_factory_user" + micro_postfix, "test_pass",
                    "test_mail@tvb.org" + micro_postfix, True, "user")
        self.user = dao.store_entity(user)

        # Now create a project
        project_service = ProjectService()
        data = dict(name='DatatypesFactoryProject' + micro_postfix, description='test_desc', users=[])
        self.project = project_service.store_project(self.user, True, None, **data)

        # Create algorithm
        alg_category = AlgorithmCategory('one', True)
        dao.store_entity(alg_category)
        ad = Algorithm(IntrospectionRegistry.SIMULATOR_MODULE, IntrospectionRegistry.SIMULATOR_CLASS,
                       alg_category.id)
        self.algorithm = dao.get_algorithm_by_module(IntrospectionRegistry.SIMULATOR_MODULE,
                                                     IntrospectionRegistry.SIMULATOR_CLASS)
        if self.algorithm is None:
            self.algorithm = dao.store_entity(ad)

        # Create an operation
        self.meta = {DataTypeMetaData.KEY_SUBJECT: self.USER_FULL_NAME,
                     DataTypeMetaData.KEY_STATE: self.DATATYPE_STATE}
        operation = Operation(self.user.id, self.project.id, self.algorithm.id, 'test parameters',
                              meta=json.dumps(self.meta), status=STATUS_FINISHED)
        self.operation = dao.store_entity(operation)

    def get_project(self):
        """
        Return project to which generated data types are assigned
        """
        return self.project

    def get_operation(self):
        """
        Return operation to which generated data types are assigned
        """
        return self.operation

    def get_user(self):
        """
        Return user to which generated data types are assigned
        """
        return self.user

    def store_datatype(self, inst=None, subject=USER_FULL_NAME, state=DATATYPE_STATE, operation_id=None):
        """
        This method creates a simple data type
        """
        inst = inst or ValueWrapperIndex()
        inst.subject = subject or self.USER_FULL_NAME
        inst.state = state or self.DATATYPE_STATE
        inst.fk_from_operation = operation_id or self.operation.id
        inst.type = inst.__class__.__name__
        inst.module = inst.__class__.__module__
        dao.store_entity(inst)
        return inst

    def __create_operation(self):
        """
        Create a operation entity. Return the operation, algo_id and the storage path.
        """
        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe", DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        algorithm = FlowService().get_algorithm_by_module_and_class(IntrospectionRegistry.SIMULATOR_MODULE,
                                                                    IntrospectionRegistry.SIMULATOR_CLASS)
        operation = Operation(self.user.id, self.project.id, algorithm.id, json.dumps(''), meta=json.dumps(meta),
                              status=STATUS_STARTED)
        operation = dao.store_entity(operation)
        storage_path = FilesHelper().get_project_folder(self.project, str(operation.id))
        return operation, algorithm.id, storage_path

    def create_connectivity(self, nodes=74):
        """
        Create a connectivity that will be used in "non-dummy" burst launches (with the actual simulator).
        """
        operation, algo_id, storage_path = self.__create_operation()
        connectivity = Connectivity(storage_path=storage_path)
        connectivity.weights = numpy.ones((nodes, nodes))
        connectivity.centres = numpy.ones((nodes, 3))
        adapter_instance = StoreAdapter([connectivity])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return algo_id, connectivity

    def create_timeseries(self, connectivity, ts_type=None, sensors=None):
        """
        Create a stored TimeSeries entity.
        """
        operation, _, storage_path = self.__create_operation()

        if ts_type == "EEG":
            time_series = TimeSeriesEEG(storage_path=storage_path, sensors=sensors)
        else:
            rm = dao.get_generic_entity(RegionMapping, connectivity.gid, '_connectivity')
            if len(rm) < 1:
                rm = None
            else:
                rm = rm[0]
            time_series = TimeSeriesRegion(storage_path=storage_path, connectivity=connectivity, region_mapping=rm)

        data = numpy.random.random((10, 10, 10, 10))
        time_series.write_data_slice(data)
        time_series.write_time_slice(numpy.arange(10))
        adapter_instance = StoreAdapter([time_series])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        time_series = dao.get_datatype_by_gid(time_series.gid)
        return time_series

    def create_stimulus(self, connectivity):
        """
        :param connectivity: Connectivity to create stimuli for its regions
        :return: persisted region Stimuli instance
        """

        operation, _, storage_path = self.__create_operation()
        stimuli_region = StimuliRegion(storage_path=storage_path)
        stimuli_region.connectivity = connectivity
        stimuli_region.weight = numpy.random.random((connectivity.number_of_regions, 1)).tolist()
        stimuli_region.temporal = PulseTrain()

        adapter_instance = StoreAdapter([stimuli_region])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return stimuli_region

    def create_covariance(self, time_series):
        """
        :returns: a stored DataType Covariance.
        """
        operation, _, storage_path = self.__create_operation()
        covariance = Covariance(storage_path=storage_path, source=time_series)
        covariance.write_data_slice(numpy.random.random((10, 10, 10)))
        adapter_instance = StoreAdapter([covariance])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return covariance

    def create_crosscoherence(self, time_series):
        """
        :returns: a stored entity of type CoherenceSpectrum
        """
        operation, _, storage_path = self.__create_operation()
        partial_coh = CoherenceSpectrum(array_data=numpy.random.random((10, 10, 10, 10)), use_storage=False)
        coherence = CoherenceSpectrum(source=time_series, storage_path=storage_path, frequency=0.1, nfft=256)
        coherence.write_data_slice(partial_coh)
        coherence.close_file()
        adapter_instance = StoreAdapter([coherence])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return coherence

    def create_crosscorrelation(self, time_series):
        """
        :returns: `CrossCorrelation` stored entity.
        """
        operation, _, storage_path = self.__create_operation()
        partial_corr = CrossCorrelation(array_data=numpy.random.random((10, 10, 10, 10, 10)), use_storage=False)
        crossc = CrossCorrelation(source=time_series, storage_path=storage_path, time=list(range(10)))
        crossc.write_data_slice(partial_corr)
        crossc.close_file()
        adapter_instance = StoreAdapter([crossc])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return crossc

    def create_surface(self):
        """
        Create a dummy surface entity.
        :returns: (Algorithm Identifier, stored Surface entity)
        """
        operation, algo_id, storage_path = self.__create_operation()
        surface = CorticalSurface(storage_path=storage_path)
        surface.vertices = numpy.array([[-10, 0, 0],
                                        [0, 0, -10],
                                        [10, 0, 0],
                                        [0, 10, 0]], dtype=float)
        surface.triangles = numpy.array([[0, 1, 2],
                                         [0, 1, 3],
                                         [1, 2, 3],
                                         [0, 2, 3]], dtype=int)
        surface.number_of_triangles = 4
        surface.number_of_vertices = 4
        surface.triangle_normals = numpy.ones((4, 3))
        surface.vertex_normals = numpy.ones((4, 3))
        surface.zero_based_triangles = True
        surface.validate()
        adapter_instance = StoreAdapter([surface])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return algo_id, surface

    def create_connectivity_measure(self, connectivity):
        """
        :returns: persisted entity ConnectivityMeasure
        """
        operation, _, storage_path = self.__create_operation()
        conn_measure = ConnectivityMeasure(storage_path=storage_path)
        conn_measure.connectivity = connectivity
        adapter_instance = StoreAdapter([conn_measure])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return conn_measure

    def create_datatype_measure(self, analyzed_entity, operation=None, storage_path=None):
        """
        :return: persisted DatatypeMeasure
        """
        if operation is None:
            operation, _, storage_path = self.__create_operation()
        measure = DatatypeMeasureIndex()
        measure.metrics = self.DATATYPE_MEASURE_METRIC
        measure.source = analyzed_entity
        adapter_instance = StoreAdapter([measure])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return measure

    def create_ica(self, timeseries):
        """
        :returns: persisted entity IndependentComponents
        """
        operation, _, storage_path = self.__create_operation()
        partial_ts = TimeSeries(use_storage=False)
        partial_ts.data = numpy.random.random((10, 10, 10, 10))
        partial_ica = IndependentComponents(source=partial_ts,
                                            component_time_series=numpy.random.random((10, 10, 10, 10)),
                                            prewhitening_matrix=numpy.random.random((10, 10, 10, 10)),
                                            unmixing_matrix=numpy.random.random((10, 10, 10, 10)),
                                            n_components=10, use_storage=False)
        ica = IndependentComponents(source=timeseries, n_components=10, storage_path=storage_path)
        ica.write_data_slice(partial_ica)
        adapter_instance = StoreAdapter([ica])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return ica

    def create_datatype_group(self, subject=USER_FULL_NAME, state=DATATYPE_STATE, ):
        """ 
        This method creates, stores and returns a DataTypeGroup entity.
        """
        group = OperationGroup(self.project.id, ranges=[json.dumps(self.RANGE_1), json.dumps(self.RANGE_2)])
        group = dao.store_entity(group)
        group_ms = OperationGroup(self.project.id, ranges=[json.dumps(self.RANGE_1), json.dumps(self.RANGE_2)])
        group_ms = dao.store_entity(group_ms)

        datatype_group = DataTypeGroup(group, subject=subject, state=state, operation_id=self.operation.id)
        # Set storage path, before setting data
        datatype_group.storage_path = self.files_helper.get_project_folder(self.project, str(self.operation.id))
        datatype_group = dao.store_entity(datatype_group)

        dt_group_ms = DataTypeGroup(group_ms, subject=subject, state=state, operation_id=self.operation.id)
        # Set storage path, before setting data
        dt_group_ms.storage_path = self.files_helper.get_project_folder(self.project, str(self.operation.id))
        dao.store_entity(dt_group_ms)

        # Now create some data types and add them to group
        for range_val1 in self.RANGE_1[1]:
            for range_val2 in self.RANGE_2[1]:
                operation = Operation(self.user.id, self.project.id, self.algorithm.id, 'test parameters',
                                      meta=json.dumps(self.meta), status=STATUS_FINISHED,
                                      range_values=json.dumps({self.RANGE_1[0]: range_val1,
                                                               self.RANGE_2[0]: range_val2}))
                operation.fk_operation_group = group.id
                operation = dao.store_entity(operation)
                datatype = self.create_datatype_with_storage(operation_id=operation.id)
                datatype.number1 = range_val1
                datatype.number2 = range_val2
                datatype.fk_datatype_group = datatype_group.id
                datatype.set_operation_id(operation.id)
                dao.store_entity(datatype)

                op_ms = Operation(self.user.id, self.project.id, self.algorithm.id, 'test parameters',
                                  meta=json.dumps(self.meta), status=STATUS_FINISHED,
                                  range_values=json.dumps({self.RANGE_1[0]: range_val1,
                                                           self.RANGE_2[0]: range_val2}))
                op_ms.fk_operation_group = group_ms.id
                op_ms = dao.store_entity(op_ms)
                self.create_datatype_measure(datatype, op_ms,
                                             FilesHelper().get_project_folder(self.project, str(op_ms.id)))

        return datatype_group
