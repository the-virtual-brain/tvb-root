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
This module contains methods for creating persisted data-types for tests.

.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""


import json
import numpy
from datetime import datetime
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.operation_service import OperationService
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.time_series import TimeSeries, TimeSeriesEEG, TimeSeriesRegion
from tvb.datatypes.graph import Covariance, ConnectivityMeasure
from tvb.datatypes.spectral import CoherenceSpectrum
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.mode_decompositions import IndependentComponents
from tvb.datatypes.mapped_values import DatatypeMeasure
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes.datatype2 import Datatype2
from tvb.tests.framework.adapters.storeadapter import StoreAdapter



class DatatypesFactory():
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
        now = datetime.now()
        micro_postfix = "_%d" % now.microsecond

        # Here create all structures needed later for data types creation
        self.files_helper = FilesHelper()

        # First create user 
        user = model.User("datatype_factory_user" + micro_postfix, "test_pass",
                          "test_mail@tvb.org" + micro_postfix, True, "user")
        self.user = dao.store_entity(user)

        # Now create a project
        project_service = ProjectService()
        data = dict(name='DatatypesFactoryProject' + micro_postfix, description='test_desc', users=[])
        self.project = project_service.store_project(self.user, True, None, **data)

        # Create algorithm
        alg_category = model.AlgorithmCategory('one', True)
        dao.store_entity(alg_category)
        alg_group = model.AlgorithmGroup("test_module1", "classname1", alg_category.id)
        dao.store_entity(alg_group)
        algorithm = model.Algorithm(alg_group.id, 'id', name='', req_data='', param_name='', output='')
        self.algorithm = dao.store_entity(algorithm)

        #Create an operation
        self.meta = {DataTypeMetaData.KEY_SUBJECT: self.USER_FULL_NAME,
                     DataTypeMetaData.KEY_STATE: self.DATATYPE_STATE}
        operation = model.Operation(self.user.id, self.project.id, self.algorithm.id, 'test parameters',
                                    meta=json.dumps(self.meta), status=model.STATUS_FINISHED,
                                    method_name=ABCAdapter.LAUNCH_METHOD)
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


    def _store_datatype(self, data_type, operation_id=None):
        """
        Launch adapter to store a create a persistent DataType.
        """
        operation_id = operation_id or self.operation.id
        data_type.type = data_type.__class__.__name__
        data_type.module = data_type.__class__.__module__
        data_type.subject = self.USER_FULL_NAME
        data_type.state = self.DATATYPE_STATE
        data_type.set_operation_id(operation_id)

        adapter_instance = StoreAdapter([data_type])
        operation = dao.get_operation_by_id(operation_id)
        OperationService().initiate_prelaunch(operation, adapter_instance, {})

        return data_type


    def create_simple_datatype(self, subject=USER_FULL_NAME, state=DATATYPE_STATE):
        """
        This method creates a simple data type
        """
        datatype_inst = Datatype1()
        self._fill_datatype(datatype_inst, subject, state)

        # Store data type
        return self._store_datatype(datatype_inst)


    def create_datatype_with_storage(self, subject=USER_FULL_NAME, state=DATATYPE_STATE,
                                     data=DATATYPE_DATA, operation_id=None):
        """
        This method creates and stores a data type which imply storage on the file system.
        """
        datatype_inst = Datatype2()
        self._fill_datatype(datatype_inst, subject, state, operation_id)

        datatype_inst.string_data = data

        return self._store_datatype(datatype_inst, operation_id)


    def _fill_datatype(self, datatype, subject, state, operation_id=None):
        """
        This method sets some common attributes on dataType 
        """
        operation_id = operation_id or self.operation.id
        datatype.subject = subject
        datatype.state = state
        # Set_operation_id also sets storage_path attribute
        datatype.set_operation_id(operation_id)


    def __create_operation(self):
        """
        Create a operation entity. Return the operation, algo_id and the storage path.
        """
        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe", DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        algorithm, algo_group = FlowService().get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
        operation = model.Operation(self.user.id, self.project.id, algo_group.id, json.dumps(''), meta=json.dumps(meta),
                                    status=model.STATUS_STARTED, method_name=ABCAdapter.LAUNCH_METHOD)
        operation = dao.store_entity(operation)
        storage_path = FilesHelper().get_project_folder(self.project, str(operation.id))
        return operation, algorithm.id, storage_path


    def create_connectivity(self):
        """
        Create a connectivity that will be used in "non-dummy" burst launches (with the actual simulator).
        """
        operation, algo_id, storage_path = self.__create_operation()
        connectivity = Connectivity(storage_path=storage_path)
        connectivity.weights = numpy.ones((74, 74))
        connectivity.centres = numpy.ones((74, 3))
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
            time_series = TimeSeriesRegion(storage_path=storage_path, connectivity=connectivity)
        data = numpy.random.random((10, 10, 10, 10))
        time = numpy.arange(10)
        time_series.write_data_slice(data)
        time_series.write_time_slice(time)
        adapter_instance = StoreAdapter([time_series])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        time_series = dao.get_datatype_by_gid(time_series.gid)
        return time_series


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
        crossc = CrossCorrelation(source=time_series, storage_path=storage_path, time=range(10))
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


    def create_datatype_measure(self, analyzed_entity):
        """
        :return: persisted DatatypeMeasure
        """
        operation, _, storage_path = self.__create_operation()
        measure = DatatypeMeasure(storage_path=storage_path, metrics=self.DATATYPE_MEASURE_METRIC)
        measure.analyzed_datatype = analyzed_entity
        adapter_instance = StoreAdapter([measure])
        OperationService().initiate_prelaunch(operation, adapter_instance, {})
        return measure


    def create_ICA(self, timeseries):
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
        group = model.OperationGroup(self.project.id, ranges=[json.dumps(self.RANGE_1), json.dumps(self.RANGE_2)])
        group = dao.store_entity(group)

        datatype_group = model.DataTypeGroup(group, subject=subject, state=state, operation_id=self.operation.id)
        # Set storage path, before setting data
        datatype_group.storage_path = self.files_helper.get_project_folder(self.project, str(self.operation.id))
        datatype_group = dao.store_entity(datatype_group)

        # Now create some data types and add them to group
        for range_val1 in self.RANGE_1[1]:
            for range_val2 in self.RANGE_2[1]:
                operation = model.Operation(self.user.id, self.project.id, self.algorithm.id, 'test parameters',
                                            meta=json.dumps(self.meta), status=model.STATUS_FINISHED,
                                            method_name=ABCAdapter.LAUNCH_METHOD,
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
                self.create_datatype_measure(datatype)

        return datatype_group