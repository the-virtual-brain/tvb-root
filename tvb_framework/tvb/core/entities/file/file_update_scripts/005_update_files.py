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
Upgrade script from H5 version 4 to version 5 (for tvb release 2.0)

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
.. creationdate:: Autumn of 2020
"""

import json
import os
import sys
import uuid
import numpy

from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter, CortexViewModel
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Range
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistory
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataTypeGroup
from tvb.core.entities.storage import dao, SA_SESSIONMAKER
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import REGISTRY
from tvb.core.neotraits.h5 import H5File, STORE_STRING, DataSetMetaData
from tvb.core.services.import_service import OPERATION_XML, ImportService, Operation2ImportData
from tvb.core.utils import date2string, string2date
from tvb.datatypes.sensors import SensorTypesEnum
from tvb.storage.h5.file.exceptions import MissingDataSetException, IncompatibleFileManagerException, \
    FileMigrationException
from tvb.storage.storage_interface import StorageInterface

LOGGER = get_logger(__name__)
FIELD_SURFACE_MAPPING = "has_surface_mapping"
FIELD_VOLUME_MAPPING = "has_volume_mapping"
DATE_FORMAT_V4_DB = '%Y-%m-%d %H:%M:%S.%f'
DATE_FORMAT_V4_H5 = 'datetime:' + DATE_FORMAT_V4_DB


# Generic parsing functions

def _parse_bool(value, storage_manager):
    value = value.lower().replace("bool:", '')
    if value in ['true', '1', 'on']:
        return storage_manager.serialize_bool(True), True
    return storage_manager.serialize_bool(False), False


def _value2str(value):
    return str(value, 'utf-8')


def _parse_gid(old_gid):
    return uuid.UUID(old_gid).urn


def _lowercase_first_character(string):
    """
    One-line function which converts the first character of a string to lowercase and
    handles empty strings and None values
    """
    return string[:1].lower() + string[1:] if string else ''


def _pop_lengths(root_metadata):
    root_metadata.pop('length_1d')
    root_metadata.pop('length_2d')
    root_metadata.pop('length_3d')
    root_metadata.pop('length_4d')

    return root_metadata


def _pop_common_metadata(root_metadata):
    root_metadata.pop('label_x')
    root_metadata.pop('label_y')
    root_metadata.pop('aggregation_functions')
    root_metadata.pop('dimensions_labels')
    root_metadata.pop('nr_dimensions')


def _bytes_ds_to_string_ds(ds_name, storage_manager):
    bytes_labels = storage_manager.get_data(ds_name)
    string_labels = []
    for i in range(len(bytes_labels)):
        string_labels.append(_value2str(bytes_labels[i]))

    storage_manager.remove_data(ds_name)
    storage_manager.store_data(numpy.asarray(string_labels).astype(STORE_STRING), ds_name)
    return storage_manager


# Specific datatype migration functions

def _migrate_connectivity(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['number_of_connections'] = int(root_metadata['number_of_connections'])
    root_metadata['number_of_regions'] = int(root_metadata['number_of_regions'])

    storage_manager = kwargs['storage_manager']
    root_metadata['undirected'], _ = _parse_bool(root_metadata['undirected'], storage_manager)

    if root_metadata['saved_selection'] == 'null':
        root_metadata['saved_selection'] = '[]'

    metadata = ['centres', 'region_labels', 'tract_lengths', 'weights']
    extra_metadata = ['orientations', 'areas', 'cortical', 'hemispheres', 'orientations']
    input_file = kwargs['input_file']
    _bytes_ds_to_string_ds('region_labels', storage_manager)

    try:
        storage_manager.get_data('hemispheres')
    except MissingDataSetException:
        # In case the Connectivity file does not hold a hemispheres dataset, write one by splitting regions in half
        all_regions = root_metadata['number_of_regions']
        right_regions = all_regions / 2
        left_regions = all_regions - right_regions
        hemispheres = int(right_regions) * [True] + int(left_regions) * [False]
        storage_manager.store_data(numpy.array(hemispheres), 'hemispheres')

    for mt in extra_metadata:
        try:
            storage_manager.get_metadata(input_file, mt)
            metadata.append(mt)
        except MissingDataSetException:
            pass

    operation_xml_parameters = kwargs['operation_xml_parameters']

    # Parsing in case Connectivity was imported via ZIPImporter
    if 'normalization' in operation_xml_parameters and operation_xml_parameters['normalization'] == 'none':
        del operation_xml_parameters['normalization']

    # Parsing in case the Connectivity was generated via ConnectivityCreator
    if 'new_weights' in operation_xml_parameters:
        operation_xml_parameters['new_weights'] = numpy.array(json.loads(operation_xml_parameters['new_weights']))
    if 'new_tracts' in operation_xml_parameters:
        operation_xml_parameters['new_tracts'] = numpy.array(json.loads(operation_xml_parameters['new_tracts']))
    if 'interest_area_indexes' in operation_xml_parameters:
        operation_xml_parameters['interest_area_indexes'] = numpy.array(
            json.loads(operation_xml_parameters['interest_area_indexes']))
    if 'is_branch' in operation_xml_parameters:
        _, operation_xml_parameters['is_branch'] = _parse_bool(operation_xml_parameters['is_branch'], storage_manager)
    if 'original_connectivity' in operation_xml_parameters:
        operation_xml_parameters['original_connectivity'] = uuid.UUID(
            operation_xml_parameters['original_connectivity'])

    # Parsing in case Connectivity was imported via CSVImporter
    if 'input_data' in operation_xml_parameters:
        operation_xml_parameters['input_data'] = uuid.UUID(operation_xml_parameters['input_data'])

    storage_manager.remove_metadata('Mean non zero', 'tract_lengths')
    storage_manager.remove_metadata('Min. non zero', 'tract_lengths')
    storage_manager.remove_metadata('Var. non zero', 'tract_lengths')
    storage_manager.remove_metadata('Mean non zero', 'weights')
    storage_manager.remove_metadata('Min. non zero', 'weights')
    storage_manager.remove_metadata('Var. non zero', 'weights')

    _migrate_dataset_metadata(metadata, storage_manager)

    return {'operation_xml_parameters': operation_xml_parameters}


def _migrate_surface(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['edge_max_length'] = float(root_metadata['edge_max_length'])
    root_metadata['edge_mean_length'] = float(root_metadata['edge_mean_length'])
    root_metadata['edge_min_length'] = float(root_metadata['edge_min_length'])
    root_metadata['number_of_split_slices'] = int(root_metadata['number_of_split_slices'])
    root_metadata['number_of_triangles'] = int(root_metadata['number_of_triangles'])
    root_metadata['number_of_vertices'] = int(root_metadata['number_of_vertices'])

    storage_manager = kwargs['storage_manager']
    root_metadata['zero_based_triangles'], _ = _parse_bool(root_metadata['zero_based_triangles'], storage_manager)
    root_metadata['bi_hemispheric'], _ = _parse_bool(root_metadata['bi_hemispheric'], storage_manager)
    root_metadata['valid_for_simulations'], _ = _parse_bool(root_metadata['valid_for_simulations'], storage_manager)

    root_metadata["surface_type"] = root_metadata["surface_type"].replace("\"", '')

    operation_xml_parameters = kwargs['operation_xml_parameters']
    _, operation_xml_parameters['zero_based_triangles'] = _parse_bool(root_metadata['zero_based_triangles'],
                                                                      storage_manager)

    if 'should_center' in operation_xml_parameters:
        _, operation_xml_parameters['should_center'] = _parse_bool(operation_xml_parameters['should_center'],
                                                                   storage_manager)

    if storage_manager.get_data('split_triangles', ignore_errors=True) is None:
        kwargs['storage_manager'].store_data([], 'split_triangles')

    _migrate_dataset_metadata(['split_triangles', 'triangle_normals', 'triangles', 'vertex_normals', 'vertices'],
                              kwargs['storage_manager'])

    return {'operation_xml_parameters': operation_xml_parameters}


def _migrate_region_mapping(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)

    root_metadata['surface'] = _parse_gid(root_metadata['surface'])
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])
    algorithm = '{"classname": "RegionMappingImporter", "module": "tvb.adapters.uploaders.region_mapping_importer"}'

    return {'algorithm': algorithm, 'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_region_volume_mapping(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)

    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])
    root_metadata['volume'] = _parse_gid(root_metadata['volume'])

    storage_manager = kwargs['storage_manager']
    array_data = storage_manager.get_data('array_data')
    array_data = array_data.astype(int)
    storage_manager.remove_data('array_data')
    storage_manager.store_data(array_data, 'array_data')

    _migrate_dataset_metadata(['array_data'], storage_manager)

    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_sensors(datasets, **kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['number_of_sensors'] = int(root_metadata['number_of_sensors'])
    root_metadata['sensors_type'] = root_metadata["sensors_type"].replace("\"", '')
    storage_manager = kwargs['storage_manager']
    root_metadata['has_orientation'], _ = _parse_bool(root_metadata['has_orientation'], storage_manager)

    input_file = kwargs['input_file']
    storage_manager.remove_metadata('Size', 'labels')
    storage_manager.remove_metadata('Size', 'locations')
    storage_manager.remove_metadata('Variance', 'locations')
    _bytes_ds_to_string_ds('labels', storage_manager)
    _migrate_dataset_metadata(datasets, storage_manager)
    algorithm = '{"classname": "SensorsImporter", "module": "tvb.adapters.uploaders.sensors_importer"}'
    kwargs['operation_xml_parameters']['sensors_file'] = input_file
    return algorithm, kwargs['operation_xml_parameters']


def _migrate_eeg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypesEnum.TYPE_EEG.value
    return {'algorithm': algorithm, 'operation_xml_parameters': operation_xml_parameters}


def _migrate_meg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations', 'orientations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypesEnum.TYPE_MEG.value
    return {'algorithm': algorithm, 'operation_xml_parameters': operation_xml_parameters}


def _migrate_seeg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypesEnum.TYPE_INTERNAL.value
    return {'algorithm': algorithm, 'operation_xml_parameters': operation_xml_parameters}


def _migrate_projection(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['sensors'] = _parse_gid(root_metadata['sensors'])
    root_metadata['sources'] = _parse_gid(root_metadata['sources'])
    root_metadata['written_by'] = "tvb.adapters.datatypes.h5.projections_h5.ProjectionMatrixH5"
    root_metadata['projection_type'] = root_metadata["projection_type"].replace("\"", '')

    storage_manager = kwargs['storage_manager']
    storage_manager.remove_metadata('Size', 'projection_data')
    storage_manager.remove_metadata('Variance', 'projection_data')

    kwargs['operation_xml_parameters']['projection_file'] = kwargs['input_file']

    _migrate_dataset_metadata(['projection_data'], storage_manager)
    algorithm = '{"classname": "ProjectionMatrixSurfaceEEGImporter", "module": ' \
                '"tvb.adapters.uploaders.projection_matrix_importer"} '

    return {'algorithm': algorithm, 'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_local_connectivity(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['cutoff'] = float(root_metadata['cutoff'])
    root_metadata['surface'] = _parse_gid(root_metadata['surface'])

    storage_manager = kwargs['storage_manager']
    matrix_metadata = storage_manager.get_metadata('matrix')
    matrix_metadata['Shape'] = _value2str(matrix_metadata['Shape'])
    matrix_metadata['dtype'] = _value2str(matrix_metadata['dtype'])
    matrix_metadata['format'] = _value2str(matrix_metadata['format'])
    storage_manager.set_metadata(matrix_metadata, 'matrix')
    operation_xml_parameters = kwargs['operation_xml_parameters']

    equation = json.loads(root_metadata['equation'])
    additional_params = None
    if equation is not None:
        equation['type'] = equation['__mapped_class']
        del equation['__mapped_class']
        del equation['__mapped_module']
        root_metadata['equation'] = json.dumps(equation)

        equation_name = operation_xml_parameters['equation']
        equation = getattr(sys.modules['tvb.datatypes.equations'],
                           equation_name)()
        operation_xml_parameters['equation'] = equation
        operation_xml_parameters['cutoff'] = float(operation_xml_parameters['cutoff'])
        parameters = {}

        for xml_param in operation_xml_parameters:
            if '_parameters_parameters_' in xml_param:
                new_key = xml_param.replace('_parameters_option_' + equation_name + '_parameters_parameters_',
                                            '.parameters.')
                param_name_start_idx = new_key.rfind('.')
                param_name = new_key[param_name_start_idx + 1:]
                parameters[param_name] = float(operation_xml_parameters[xml_param])
        additional_params = [['equation', 'parameters', parameters]]

    operation_xml_parameters['surface'] = root_metadata['surface']

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_connectivity_annotations(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])
    root_metadata['written_by'] = "tvb.adapters.datatypes.h5.annotation_h5.ConnectivityAnnotationsH5"

    _migrate_dataset_metadata(['region_annotations'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_monitors(operation_xml_parameters, model):
    monitors = operation_xml_parameters['monitors']
    hrf_kernel_name = None
    bold_index = None
    for i in range(len(monitors)):
        monitor_name = operation_xml_parameters['monitors'][i]
        monitor = getattr(sys.modules['tvb.core.entities.file.simulator.view_model'],
                          monitor_name + 'ViewModel')()
        operation_xml_parameters['monitors'][i] = monitor
        if monitor_name != 'Raw':
            monitor.period = float(operation_xml_parameters['monitors_parameters_option_' + monitor_name + '_period'])
            variables_of_interest = eval(operation_xml_parameters['monitors_parameters_option_' + monitor_name +
                                                                  '_variables_of_interest'])
            if len(variables_of_interest) != 0:
                monitor.variables_of_interest = numpy.array(variables_of_interest, dtype=numpy.int64)
            else:
                monitor.variables_of_interest = numpy.array([i for i in range(len(model.variables_of_interest))])
            if monitor_name == 'SpatialAverage':
                spatial_mask = eval(operation_xml_parameters['monitors_parameters_option_SpatialAverage_spatial_mask'])
                if len(spatial_mask) != 0:
                    monitor.spatial_mask = numpy.array(spatial_mask, dtype=numpy.int64)
            if monitor_name in ['EEG', 'MEG', 'iEEG']:
                _set_sensors_view_model_attributes(operation_xml_parameters, monitor_name, i)

            if monitor_name == 'Bold':
                hrf_kernel_name = operation_xml_parameters.pop('monitors_parameters_option_Bold_hrf_kernel')
                hrf_kernel = getattr(sys.modules['tvb.datatypes.equations'], hrf_kernel_name)
                monitor.hrf_kernel = hrf_kernel()
                bold_index = i

                operation_xml_parameters.pop('monitors_parameters_option_Bold_hrf_kernel_parameters_option_' +
                                             hrf_kernel_name + '_equation')

    return hrf_kernel_name, bold_index


def _migrate_noise(operation_xml_parameters, integrator, integrator_name, noise_type_name):
    equation_name = None
    noise_type = getattr(sys.modules['tvb.core.entities.file.simulator.view_model'],
                         noise_type_name + 'NoiseViewModel')
    integrator.noise = noise_type()

    integrator_param_base_name = 'integrator_parameters_option_' + integrator_name + '_noise_' + \
                                 'parameters_option_' + noise_type_name + '_'
    integrator.noise.ntau = float(operation_xml_parameters.pop(integrator_param_base_name + 'ntau'))
    integrator.noise.noise_seed = int(operation_xml_parameters.pop(integrator_param_base_name +
                                                                   'random_stream_parameters_option_RandomStream' + \
                                                                   '_init_seed'))
    integrator.noise.nsig = numpy.array(eval(operation_xml_parameters.pop(integrator_param_base_name + 'nsig')),
                                        dtype=numpy.float64)
    operation_xml_parameters.pop(integrator_param_base_name + 'random_stream')

    replace_for_branch = 'Multiplicative'
    if noise_type_name == 'Multiplicative':
        replace_for_branch = 'Additive'
        equation_name = operation_xml_parameters.pop(integrator_param_base_name + 'b')
        equation_type = getattr(sys.modules['tvb.datatypes.equations'], equation_name)
        integrator.noise.b = equation_type()
        operation_xml_parameters.pop(integrator_param_base_name + 'b_parameters_option_' +
                                     equation_name + '_equation')

    operation_xml_parameters.pop((integrator_param_base_name + 'random_stream_parameters_option_RandomStream' +
                                  '_init_seed').replace(noise_type_name, replace_for_branch), None)
    operation_xml_parameters.pop((integrator_param_base_name + 'ntau').replace(noise_type_name,
                                                                               replace_for_branch), None)
    operation_xml_parameters.pop((integrator_param_base_name + 'random_stream').replace(noise_type_name,
                                                                                        replace_for_branch), None)
    operation_xml_parameters.pop((integrator_param_base_name + 'nsig').replace(noise_type_name,
                                                                               replace_for_branch), None)

    return integrator_param_base_name, equation_name


def _migrate_common_root_metadata_time_series(metadata, root_metadata, storage_manager, input_file):
    root_metadata.pop(FIELD_SURFACE_MAPPING, False)
    root_metadata.pop(FIELD_VOLUME_MAPPING, False)
    _pop_lengths(root_metadata)

    root_metadata['sample_period'] = float(root_metadata['sample_period'])
    root_metadata['start_time'] = float(root_metadata['start_time'])
    root_metadata['user_tag_1'] = ''

    root_metadata["sample_period_unit"] = root_metadata["sample_period_unit"].replace("\"", '')
    root_metadata[DataTypeMetaData.KEY_TITLE] = root_metadata[DataTypeMetaData.KEY_TITLE].replace("\"", '')
    _migrate_dataset_metadata(metadata, storage_manager)


def _migrate_time_series(operation_xml_parameters):
    operation_xml_parameters.pop('', None)
    if 'surface' in operation_xml_parameters and operation_xml_parameters['surface'] == '':
        operation_xml_parameters['surface'] = None
    if 'stimulus' in operation_xml_parameters and operation_xml_parameters['stimulus'] == '':
        operation_xml_parameters['stimulus'] = None

    if len(operation_xml_parameters) == 0:
        return operation_xml_parameters, None

    coupling_name = operation_xml_parameters['coupling']
    coupling = getattr(sys.modules['tvb.simulator.coupling'], coupling_name)()
    operation_xml_parameters['coupling'] = coupling

    model_name = operation_xml_parameters['model']
    model = getattr(sys.modules['tvb.simulator.models'], model_name[0].upper() + model_name[1:])()
    operation_xml_parameters['model'] = model

    integrator_name = operation_xml_parameters['integrator']
    integrator = getattr(sys.modules['tvb.core.entities.file.simulator.view_model'],
                         integrator_name + 'ViewModel')()
    operation_xml_parameters['integrator'] = integrator

    operation_xml_parameters['conduction_speed'] = float(operation_xml_parameters['conduction_speed'])
    operation_xml_parameters['simulation_length'] = float(operation_xml_parameters['simulation_length'])

    hrf_kernel_name, bold_index = _migrate_monitors(operation_xml_parameters, model)

    noise_param_name = 'integrator_parameters_option_' + integrator_name + '_noise'
    noise_type_name = operation_xml_parameters.pop(noise_param_name, None)

    if noise_type_name is not None:
        integrator_param_base_name, equation_name = \
            _migrate_noise(operation_xml_parameters, integrator, integrator_name, noise_type_name)

    additional_params = []
    for xml_param in operation_xml_parameters:
        if 'coupling_parameters' in xml_param:
            new_param = xml_param.replace('coupling_parameters_option_' + coupling_name + '_', '')
            coupling_param = numpy.asarray(eval(str(operation_xml_parameters[xml_param])),
                                           dtype=getattr(coupling, new_param).dtype)
            additional_params.append(['coupling', new_param, coupling_param])

        elif 'model_parameters_option_' in xml_param and 'range_parameters' not in xml_param:
            new_param = xml_param.replace('model_parameters_option_' + model_name + '_', '')
            list_param = operation_xml_parameters[xml_param]

            if model_name + '_variables_of_interest' not in xml_param:
                try:
                    list_param = numpy.asarray(eval(list_param.replace(' ', ', ')),
                                               dtype=getattr(model, new_param).dtype)
                except AttributeError:
                    list_param = numpy.asarray(list_param)

            additional_params.append(['model', new_param, list_param])
        elif 'noise_parameters' in xml_param:
            new_param_name = xml_param.replace(integrator_param_base_name +
                                               'b_parameters_option_' + equation_name +
                                               '_parameters_parameters_', '')
            integrator.noise.b.parameters[new_param_name] = float(operation_xml_parameters[xml_param])

        elif 'integrator_parameters_option_' in xml_param:
            new_param = xml_param.replace('integrator_parameters_option_' + integrator_name + '_', '')
            additional_params.append(['integrator', new_param, float(operation_xml_parameters[xml_param])])
        elif 'hrf_kernel' in xml_param:
            new_param = xml_param.replace('monitors_parameters_option_Bold_hrf_kernel_parameters_option_' +
                                          hrf_kernel_name + '_parameters_parameters_', '')
            operation_xml_parameters['monitors'][bold_index].hrf_kernel.parameters[new_param] = \
                float(operation_xml_parameters[xml_param])

    return operation_xml_parameters, additional_params


def _migrate_time_series_simple(**kwargs):
    operation_xml_parameters = kwargs['operation_xml_parameters']
    root_metadata = kwargs['root_metadata']
    _migrate_common_root_metadata_time_series(['data', 'time'], root_metadata, kwargs['storage_manager'],
                                              kwargs['input_file'])

    operation_xml_parameters, additional_params = _migrate_time_series(operation_xml_parameters)
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _parse_fmri_ballon_adapter_operation(operation_xml_parameters):
    operation_xml_parameters['dt'] = float(operation_xml_parameters['dt'])
    return operation_xml_parameters, None


def _parse_mat_importer_operation(operation_xml_parameters, storage_manager):
    if 'transpose' in operation_xml_parameters:
        _, operation_xml_parameters['transpose'] = _parse_bool(operation_xml_parameters['transpose'], storage_manager)
    if 'slice' in operation_xml_parameters:
        _, operation_xml_parameters['slice'] = _parse_bool(operation_xml_parameters['slice'], storage_manager)
    if 'sampling_rate' in operation_xml_parameters:
        operation_xml_parameters['sampling_rate'] = int(operation_xml_parameters['sampling_rate'])
    operation_xml_parameters['start_time'] = int(operation_xml_parameters['start_time'])
    operation_xml_parameters['datatype'] = uuid.UUID(operation_xml_parameters['datatype'])
    return operation_xml_parameters, None


def _migrate_time_series_region(**kwargs):
    operation_xml_parameters = kwargs['operation_xml_parameters']
    root_metadata = kwargs['root_metadata']

    _migrate_common_root_metadata_time_series(['data', 'time'], root_metadata, kwargs['storage_manager'],
                                              kwargs['input_file'])
    if 'region_mapping' in root_metadata:
        root_metadata['region_mapping'] = _parse_gid(root_metadata['region_mapping'])
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])

    if 'data_file' in operation_xml_parameters:
        operation_xml_parameters, additional_params = _parse_mat_importer_operation(operation_xml_parameters,
                                                                                    kwargs['storage_manager'])
    elif 'RBM' in operation_xml_parameters:
        operation_xml_parameters, additional_params = _parse_fmri_ballon_adapter_operation(operation_xml_parameters)
    else:
        operation_xml_parameters, additional_params = _migrate_time_series(operation_xml_parameters)

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_time_series_surface(**kwargs):
    operation_xml_parameters = kwargs['operation_xml_parameters']
    root_metadata = kwargs['root_metadata']
    _migrate_common_root_metadata_time_series(['data', 'time'], root_metadata, kwargs['storage_manager'],
                                              kwargs['input_file'])

    if 'data_file' in operation_xml_parameters:
        # Parsing in case TimeSeriesSurface was imported via GiftiImporter
        operation_xml_parameters['surface'] = uuid.UUID(operation_xml_parameters['surface'])
        additional_params = None
    else:
        operation_xml_parameters, additional_params = _migrate_time_series(operation_xml_parameters)
        surface_gid = operation_xml_parameters['surface']
        cortical_surface = CortexViewModel()
        cortical_surface.surface_gid = uuid.UUID(surface_gid)
        cortical_surface.region_mapping_data = uuid.UUID(
            operation_xml_parameters['surface_parameters_region_mapping_data'])

        if len(operation_xml_parameters['surface_parameters_local_connectivity']) > 0:
            cortical_surface.local_connectivity = uuid.UUID(
                operation_xml_parameters['surface_parameters_local_connectivity'])

        cortical_surface.coupling_strength = numpy.asarray(
            eval(operation_xml_parameters['surface_parameters_coupling_strength'].replace(' ', ', ')))
        operation_xml_parameters['surface'] = cortical_surface

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _set_sensors_view_model_attributes(operation_xml_parameters, sensors_type, index):
    eeg_reference = 'monitors_parameters_option_EEG_reference'
    if eeg_reference in operation_xml_parameters:
        setattr(operation_xml_parameters['monitors'][index], 'reference', operation_xml_parameters[eeg_reference])

    setattr(operation_xml_parameters['monitors'][index], 'region_mapping',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_region_mapping'])
    setattr(operation_xml_parameters['monitors'][index], 'sensors',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_sensors'])
    setattr(operation_xml_parameters['monitors'][index], 'projection',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_projection'])


def _migrate_time_series_sensors(**kwargs):
    operation_xml_parameters = kwargs['operation_xml_parameters']
    root_metadata = kwargs['root_metadata']
    _migrate_common_root_metadata_time_series(['data', 'time'], root_metadata, kwargs['storage_manager'],
                                              kwargs['input_file'])

    if 'data_file' in operation_xml_parameters:
        # Parsing in case TimeSeriesSensors was imported via EEGMatImporter
        operation_xml_parameters['datatype'] = uuid.UUID(operation_xml_parameters['datatype'])
        additional_params = None
    else:
        operation_xml_parameters, additional_params = _migrate_time_series(operation_xml_parameters)
        root_metadata['sensors'] = _parse_gid(root_metadata['sensors'])

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_nifti_importer_operation(operation_xml_parameters, storage_manager):
    if 'apply_corrections' in operation_xml_parameters:
        _, operation_xml_parameters['apply_corrections'] = _parse_bool(operation_xml_parameters['apply_corrections'],
                                                                       storage_manager)
    if 'connectivity' in operation_xml_parameters:
        operation_xml_parameters['connectivity'] = uuid.UUID(operation_xml_parameters['connectivity'])
    return operation_xml_parameters, None


def _migrate_time_series_volume(**kwargs):
    operation_xml_parameters = kwargs['operation_xml_parameters']
    root_metadata = kwargs['root_metadata']
    _migrate_common_root_metadata_time_series(['data'], root_metadata, kwargs['storage_manager'],
                                              kwargs['input_file'])

    if 'data_file' in operation_xml_parameters:
        operation_xml_parameters, additional_params = _migrate_nifti_importer_operation(operation_xml_parameters,
                                                                                        kwargs['storage_manager'])
    else:
        operation_xml_parameters, additional_params = _migrate_time_series(operation_xml_parameters)

    root_metadata['nr_dimensions'] = int(root_metadata['nr_dimensions'])
    root_metadata['sample_rate'] = float(root_metadata['sample_rate'])
    root_metadata['volume'] = _parse_gid(root_metadata['volume'])
    # root_metadata['volume'] = root_metadata['volume']
    root_metadata.pop('nr_dimensions')
    root_metadata.pop('sample_rate')
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_volume(**kwargs):
    root_metadata = kwargs['root_metadata']
    kwargs['root_metadata']['voxel_unit'] = root_metadata['voxel_unit'].replace("\"", '')

    operation_xml_parameters = kwargs['operation_xml_parameters']
    if 'connectivity' in operation_xml_parameters and operation_xml_parameters['connectivity'] == '':
        operation_xml_parameters['connectivity'] = None

    if 'apply_corrections' in operation_xml_parameters:
        _, operation_xml_parameters['apply_corrections'] = _parse_bool(operation_xml_parameters['apply_corrections'],
                                                                       kwargs['storage_manager'])

    _migrate_dataset_metadata(['origin', 'voxel_size'], kwargs['storage_manager'])
    return {'operation_xml_parameters': operation_xml_parameters}


def _migrate_structural_mri(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)

    root_metadata['volume'] = _parse_gid(root_metadata['volume'])

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_complex_coherence_spectrum(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['epoch_length'] = float(root_metadata['epoch_length'])
    root_metadata['segment_length'] = float(root_metadata['segment_length'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])
    root_metadata['windowing_function'] = root_metadata['windowing_function'].replace("\"", '')
    root_metadata['source'] = _parse_gid(root_metadata['source'])

    _migrate_dataset_metadata(['array_data', 'cross_spectrum'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_wavelet_coefficients(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['q_ratio'] = float(root_metadata['q_ratio'])
    root_metadata['sample_period'] = float(root_metadata['sample_period'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])
    root_metadata['mother'] = root_metadata['mother'].replace("\"", '')
    root_metadata['normalisation'] = root_metadata['normalisation'].replace("\"", '')

    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['input_data'] = root_metadata['source']

    range = Range(lo=float(operation_xml_parameters['frequencies_parameters_option_Range_lo']),
                  hi=float(operation_xml_parameters['frequencies_parameters_option_Range_hi']),
                  step=float(operation_xml_parameters['frequencies_parameters_option_Range_step']))
    operation_xml_parameters['frequencies'] = range
    operation_xml_parameters['sample_period'] = float(operation_xml_parameters['sample_period'])
    operation_xml_parameters['q_ratio'] = float(operation_xml_parameters['q_ratio'])
    _migrate_dataset_metadata(['amplitude', 'array_data', 'frequencies', 'phase', 'power'],
                              kwargs['storage_manager'])
    return {'operation_xml_parameters': operation_xml_parameters}


def _migrate_coherence_spectrum(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    kwargs['root_metadata'].pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['nfft'] = int(root_metadata['nfft'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])

    storage_manager = kwargs['storage_manager']
    array_data = storage_manager.get_data('array_data')
    storage_manager.remove_data('array_data')
    storage_manager.store_data(numpy.asarray(array_data, dtype=numpy.float64), 'array_data')
    kwargs['operation_xml_parameters']['nfft'] = int(kwargs['operation_xml_parameters']['nfft'])

    _migrate_dataset_metadata(['array_data', 'frequency'], storage_manager)
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_cross_correlation(**kwargs):
    root_metadata = kwargs['root_metadata']
    kwargs['operation_xml_parameters']['datatype'] = root_metadata['source']
    root_metadata['source'] = _parse_gid(root_metadata['source'])

    root_metadata['source'] = _parse_gid(root_metadata['source'])
    _migrate_dataset_metadata(['array_data', 'time'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_fcd(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['sp'] = float(root_metadata['sp'])
    root_metadata['sw'] = float(root_metadata['sw'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['sp'] = root_metadata['sp']
    operation_xml_parameters['sw'] = root_metadata['sw']

    storage_manager = kwargs['storage_manager']
    storage_manager.set_metadata(root_metadata)

    _migrate_dataset_metadata(['array_data'], storage_manager)

    return {'operation_xml_parameters': operation_xml_parameters}


def _migrate_fourier_spectrum(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['segment_length'] = float(root_metadata['segment_length'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['input_data'] = root_metadata['source']

    _, operation_xml_parameters['detrend'] = _parse_bool(operation_xml_parameters['detrend'], kwargs['storage_manager'])
    operation_xml_parameters['segment_length'] = root_metadata['segment_length']

    _migrate_dataset_metadata(['amplitude', 'array_data', 'average_power', 'normalised_average_power',
                               'phase', 'power'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_independent_components(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['n_components'] = int(root_metadata['n_components'])
    root_metadata['source'] = _parse_gid(root_metadata['source'])

    _migrate_dataset_metadata(['component_time_series', 'mixing_matrix', 'norm_source',
                               'normalised_component_time_series', 'prewhitening_matrix',
                               'unmixing_matrix'], kwargs['storage_manager'])
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['n_components'] = int(operation_xml_parameters['n_components'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_correlation_coefficients(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_lengths(root_metadata)
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)

    root_metadata['source'] = _parse_gid(root_metadata['source'])

    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['datatype'] = root_metadata['source']
    operation_xml_parameters['t_start'] = float(operation_xml_parameters['t_start'])
    operation_xml_parameters['t_end'] = float(operation_xml_parameters['t_end'])

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_principal_components(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['source'] = _parse_gid(root_metadata['source'])

    _migrate_dataset_metadata(['component_time_series', 'fractions', 'norm_source', 'normalised_component_time_series',
                               'weights'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_covariance(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_common_metadata(root_metadata)
    root_metadata.pop(DataTypeMetaData.KEY_TITLE)
    _pop_lengths(root_metadata)

    root_metadata['source'] = _parse_gid(root_metadata['source'])

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_connectivity_measure(**kwargs):
    root_metadata = kwargs['root_metadata']
    _pop_common_metadata(root_metadata)
    _pop_lengths(kwargs['root_metadata'])
    kwargs['root_metadata']['connectivity'] = _parse_gid(root_metadata['connectivity'])

    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['data_file'] = kwargs['input_file']
    operation_xml_parameters['connectivity'] = root_metadata['connectivity']
    root_metadata['title'] = root_metadata['title'].replace('\\n', '').replace('"', '')

    storage_manager = kwargs['storage_manager']
    input_file = kwargs['input_file']

    _migrate_dataset_metadata(['array_data'], storage_manager)

    if 'parent_burst' in root_metadata or 'time_series' not in operation_xml_parameters:
        return {'operation_xml_parameters': operation_xml_parameters}

    # If we encounter a Connectivity Measure we expect that it could have more ConnectivityMeasure's and FCDs
    # so we can only set here the parent_burst attribute on all files
    folder_name = os.path.dirname(input_file)
    for file_name in os.listdir(folder_name):
        if file_name != OPERATION_XML and file_name not in input_file:
            root_metadata = storage_manager.get_metadata()
            _set_parent_burst(operation_xml_parameters['time_series'], root_metadata, storage_manager, True)

    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_datatype_measure(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['written_by'] = 'tvb.core.entities.file.simulator.datatype_measure_h5.DatatypeMeasureH5'
    operation_xml_parameters = kwargs['operation_xml_parameters']

    if 'start_point' in operation_xml_parameters:
        operation_xml_parameters['start_point'] = float(operation_xml_parameters['start_point'])
        operation_xml_parameters['segment'] = int(operation_xml_parameters['segment'])
    if 'algorithms' in operation_xml_parameters:
        operation_xml_parameters['algorithms'] = [operation_xml_parameters['algorithms']]

    root_metadata['user_tag_1'] = ''
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_stimuli_equation_params(operation_xml_parameters, equation_type):
    equation_name = operation_xml_parameters[equation_type]
    equation = getattr(sys.modules['tvb.datatypes.equations'], equation_name)()
    operation_xml_parameters[equation_type] = equation

    param_dict = {}
    for xml_param in operation_xml_parameters:
        if 'parameters_parameters' in xml_param:
            temporal_param = float(operation_xml_parameters[xml_param])
            temporal_param_name = xml_param.replace(equation_type + '_parameters_option_' + equation_name + \
                                                    '_parameters_parameters_', '')
            param_dict[temporal_param_name] = temporal_param
    return [equation_type, 'parameters', param_dict]


def _migrate_stimuli(datasets, root_metadata, storage_manager, input_file):
    _migrate_one_stimuli_param(root_metadata, 'spatial')
    _migrate_one_stimuli_param(root_metadata, 'temporal')

    for dataset in datasets:
        weights = json.loads(root_metadata[dataset])
        storage_manager.store_data(weights, dataset)
        _migrate_dataset_metadata([dataset], storage_manager)
        root_metadata.pop(dataset)


def _migrate_stimuli_region(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])
    operation_xml_parameters = kwargs['operation_xml_parameters']

    operation_xml_parameters['weight'] = numpy.asarray(json.loads(root_metadata['weight']), dtype=numpy.float64)
    _migrate_stimuli(['weight'], root_metadata, kwargs['storage_manager'], kwargs['input_file'])
    additional_params = [_migrate_stimuli_equation_params(operation_xml_parameters, 'temporal')]

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_stimuli_surface(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['surface'] = _parse_gid(root_metadata['surface'])
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['focal_points_triangles'] = numpy.asarray(
        json.loads(root_metadata['focal_points_triangles']), dtype=numpy.int_)
    _migrate_stimuli(['focal_points_surface', 'focal_points_triangles'], root_metadata, kwargs['storage_manager'],
                     kwargs['input_file'])
    additional_params = [_migrate_stimuli_equation_params(operation_xml_parameters, 'temporal'),
                         _migrate_stimuli_equation_params(operation_xml_parameters, 'spatial')]

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_value_wrapper(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['data_type'] = root_metadata['data_type'].replace("\"", '')
    root_metadata['data_name'] = root_metadata['data_name'].replace("\"", '')
    root_metadata['written_by'] = "tvb.adapters.datatypes.h5.mapped_value_h5.ValueWrapperH5"
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_simulation_state(**kwargs):
    os.remove(kwargs['input_file'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_tracts(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['region_volume_map'] = _parse_gid(root_metadata['region_volume_map'])
    _migrate_dataset_metadata(['tract_region', 'tract_start_idx', 'vertices'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_dataset_metadata(dataset_list, storage_manager):
    for dataset in dataset_list:
        metadata = DataSetMetaData.from_array(storage_manager.get_data(dataset)).to_dict()
        storage_manager.set_metadata(metadata, dataset)
        metadata = storage_manager.get_metadata(dataset)
        if 'Variance' in metadata:
            storage_manager.remove_metadata('Variance', dataset)
        if 'Size' in metadata:
            storage_manager.remove_metadata('Size', dataset)


def _migrate_one_stimuli_param(root_metadata, param_name):
    param = json.loads(root_metadata[param_name])
    new_param = dict()
    new_param['type'] = param['__mapped_class']
    new_param['parameters'] = param['parameters']
    root_metadata[param_name] = json.dumps(new_param)


def _set_parent_burst(time_series_gid, root_metadata, storage_manager, input_file, is_ascii=False):
    ts = dao.get_datatype_by_gid(uuid.UUID(time_series_gid).hex)
    if ts.fk_parent_burst:
        burst_gid = _parse_gid(ts.fk_parent_burst)
        if is_ascii:
            burst_gid = burst_gid.encode('ascii', 'ignore')
        root_metadata['parent_burst'] = burst_gid
        storage_manager.set_metadata(root_metadata)
    return ts.fk_parent_burst, ts.fk_datatype_group


def _migrate_operation_group(op_group_id, model):
    operation_group = dao.get_operationgroup_by_id(op_group_id)
    operation_group.name = operation_group.name.replace(
        '_parameters_option_{}_'.format(model.__class__.__name__), '.')
    return operation_group


def _migrate_datatype_group(operation_group, burst_gid, generic_attributes):
    datatype_group = DataTypeGroup(operation_group)
    datatype_group.fk_parent_burst = burst_gid
    datatype_group.count_results = len(dao.get_operations_in_group(operation_group.id))
    datatype_group.fill_from_generic_attributes(generic_attributes)
    stored_datatype_group = dao.store_entity(datatype_group)

    first_datatype_group_op = dao.get_operations_in_group(stored_datatype_group.fk_operation_group,
                                                          only_first_operation=True)
    stored_datatype_group.fk_from_operation = first_datatype_group_op.id
    dao.store_entity(stored_datatype_group)


datatypes_to_be_migrated = {
    'Connectivity': _migrate_connectivity,
    'BrainSkull': _migrate_surface,
    'CorticalSurface': _migrate_surface,
    'SkinAir': _migrate_surface,
    'SkullSkin': _migrate_surface,
    'EEGCap': _migrate_surface,
    'FaceSurface': _migrate_surface,
    'WhiteMatterSurface': _migrate_surface,
    'RegionMapping': _migrate_region_mapping,
    'RegionVolumeMapping': _migrate_region_volume_mapping,
    'SensorsEEG': _migrate_eeg_sensors,
    'SensorsMEG': _migrate_meg_sensors,
    'SensorsInternal': _migrate_seeg_sensors,
    'ProjectionSurfaceEEG': _migrate_projection,
    'ProjectionSurfaceMEG': _migrate_projection,
    'ProjectionSurfaceSEEG': _migrate_projection,
    'LocalConnectivity': _migrate_local_connectivity,
    'ConnectivityAnnotations': _migrate_connectivity_annotations,
    'TimeSeries': _migrate_time_series_simple,
    'TimeSeriesRegion': _migrate_time_series_region,
    'TimeSeriesSurface': _migrate_time_series_surface,
    'TimeSeriesEEG': _migrate_time_series_sensors,
    'TimeSeriesMEG': _migrate_time_series_sensors,
    'TimeSeriesSEEG': _migrate_time_series_sensors,
    'TimeSeriesVolume': _migrate_time_series_volume,
    'Volume': _migrate_volume,
    'StructuralMRI': _migrate_structural_mri,
    'ComplexCoherenceSpectrum': _migrate_complex_coherence_spectrum,
    'WaveletCoefficients': _migrate_wavelet_coefficients,
    'CoherenceSpectrum': _migrate_coherence_spectrum,
    'CrossCorrelation': _migrate_cross_correlation,
    'Fcd': _migrate_fcd,
    'FourierSpectrum': _migrate_fourier_spectrum,
    'IndependentComponents': _migrate_independent_components,
    'CorrelationCoefficients': _migrate_correlation_coefficients,
    'PrincipalComponents': _migrate_principal_components,
    'Covariance': _migrate_covariance,
    'ConnectivityMeasure': _migrate_connectivity_measure,
    'DatatypeMeasure': _migrate_datatype_measure,
    'StimuliRegion': _migrate_stimuli_region,
    'StimuliSurface': _migrate_stimuli_surface,
    'ValueWrapper': _migrate_value_wrapper,
    'SimulationState': _migrate_simulation_state,
    'Tracts': _migrate_tracts
}


def _migrate_general_part(input_file):
    # Obtain storage manager and metadata
    storage_manager = StorageInterface.get_storage_manager(input_file)
    root_metadata = storage_manager.get_metadata()

    if DataTypeMetaData.KEY_CLASS_NAME not in root_metadata:
        raise IncompatibleFileManagerException("File %s received for upgrading 4 -> 5 is not valid, due to missing "
                                               "metadata: %s" % (input_file, DataTypeMetaData.KEY_CLASS_NAME))

    # In the new format all metadata has the 'TVB_%' format, where '%' starts with a lowercase letter
    lowercase_keys = []
    for key, value in root_metadata.items():
        try:
            root_metadata[key] = _value2str(value)
        except TypeError:
            pass
        lowercase_keys.append(_lowercase_first_character(key))
        storage_manager.remove_metadata(key)

    # Update DATA_VERSION
    root_metadata = dict(zip(lowercase_keys, list(root_metadata.values())))
    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION

    # UPDATE CREATION DATE
    root_metadata[DataTypeMetaData.KEY_DATE] = date2string(
        string2date(root_metadata[DataTypeMetaData.KEY_DATE], date_format=DATE_FORMAT_V4_H5))

    # OBTAIN THE MODULE (for a few data types the old module doesn't exist anymore, in those cases the attr
    # will be set later
    try:
        datatype_class = getattr(sys.modules[root_metadata["module"]],
                                 root_metadata["type"])
        h5_class = REGISTRY.get_h5file_for_datatype(datatype_class)
        root_metadata[H5File.KEY_WRITTEN_BY] = h5_class.__module__ + '.' + h5_class.__name__
    except KeyError:
        pass

    # Other general modifications
    root_metadata['gid'] = _parse_gid(root_metadata['gid'])
    root_metadata.pop("module")
    root_metadata.pop('data_version')

    return root_metadata, storage_manager


def _get_burst_for_migration(burst_id, burst_match_dict, date_format, selected_db):
    """
    This method is supposed to only be used when migrating from version 4 to version 5.
    It finds a BurstConfig in the old format (when it did not inherit from HasTraitsIndex), deletes it
    and returns its parameters.
    """
    session = SA_SESSIONMAKER()
    burst_params = session.execute("""SELECT * FROM "BURST_CONFIGURATION" WHERE id = """ + burst_id).fetchone()
    session.close()

    if burst_params is None:
        return None, False

    burst_params_dict = {'datatypes_number': burst_params['datatypes_number'],
                         'dynamic_ids': burst_params['dynamic_ids'], 'range_1': burst_params['range1'],
                         'range_2': burst_params['range2'], 'fk_project': burst_params['fk_project'],
                         'name': burst_params['name'], 'status': burst_params['status'],
                         'error_message': burst_params['error_message'], 'start_time': burst_params['start_time'],
                         'finish_time': burst_params['finish_time'], 'fk_simulation': burst_params['fk_simulation'],
                         'fk_operation_group': burst_params['fk_operation_group'],
                         'fk_metric_operation_group': burst_params['fk_metric_operation_group']}

    if selected_db == 'sqlite':
        burst_params_dict['start_time'] = string2date(burst_params_dict['start_time'], date_format=date_format)
        burst_params_dict['finish_time'] = string2date(burst_params_dict['finish_time'], date_format=date_format)

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


def update(input_file, burst_match_dict):
    """
    :param input_file: the file that needs to be converted to a newer file storage version.
    """
    if not os.path.isfile(input_file):
        raise IncompatibleFileManagerException("Not yet implemented update for file %s" % input_file)

    # The first step is to check based on the path if this function call is part of the migration of the whole storage
    # folder or just one datatype (in the first case the second to last element in the path is a number
    split_path = input_file.split(os.path.sep)
    storage_migrate = True
    try:
        op_id = int(split_path[-2])

        # Change file names only for storage migration
        file_basename = os.path.basename(input_file)
        replaced_basename = file_basename.replace('-', '')
        replaced_basename = replaced_basename.replace('BrainSkull', 'Surface')
        replaced_basename = replaced_basename.replace('CorticalSurface', 'Surface')
        replaced_basename = replaced_basename.replace('SkinAir', 'Surface')
        replaced_basename = replaced_basename.replace('BrainSkull', 'Surface')
        replaced_basename = replaced_basename.replace('SkullSkin', 'Surface')
        replaced_basename = replaced_basename.replace('EEGCap', 'Surface')
        replaced_basename = replaced_basename.replace('FaceSurface', 'Surface')
        replaced_basename = replaced_basename.replace('SensorsEEG', 'Sensors')
        replaced_basename = replaced_basename.replace('SensorsMEG', 'Sensors')
        replaced_basename = replaced_basename.replace('SensorsInternal', 'Sensors')
        replaced_basename = replaced_basename.replace('ProjectionSurfaceEEG', 'ProjectionMatrix')
        replaced_basename = replaced_basename.replace('ProjectionSurfaceMEG', 'ProjectionMatrix')
        replaced_basename = replaced_basename.replace('ProjectionSurfaceSEEG', 'ProjectionMatrix')
        new_file_path = os.path.join(os.path.dirname(input_file), replaced_basename)
        os.rename(input_file, new_file_path)
        input_file = new_file_path
    except ValueError:
        op_id = None
        storage_migrate = False

    folder, file_name = os.path.split(input_file)
    root_metadata, storage_manager = _migrate_general_part(input_file)
    class_name = root_metadata['type']
    root_metadata.pop("type")

    files_in_folder = os.listdir(folder)
    import_service = ImportService()
    algorithm = ''

    operation_xml_parameters = {}  # params that are needed for the view_model but are not in the Operation.xml file
    additional_params = None  # params that can't be jsonified with json.dumps
    has_vm = False
    operation = None

    try:
        # Take information out from the Operation.xml file
        if op_id is not None and OPERATION_XML in files_in_folder:
            operation_file_path = os.path.join(folder, OPERATION_XML)
            operation = dao.get_operation_by_id(op_id)
            xml_operation, operation_xml_parameters, algorithm = \
                import_service.build_operation_from_file(operation.project, operation_file_path)
            try:
                operation_xml_parameters = json.loads(operation_xml_parameters)
            except NameError:
                operation_xml_parameters = operation_xml_parameters.replace('null', '\"null\"')
                operation_xml_parameters = json.load(operation_xml_parameters)
        else:
            has_vm = True

        # Calls the specific method for the current h5 class
        params = datatypes_to_be_migrated[class_name](root_metadata=root_metadata,
                                                      storage_manager=storage_manager,
                                                      operation_xml_parameters=operation_xml_parameters,
                                                      input_file=input_file)

        operation_xml_parameters = params['operation_xml_parameters']
        if 'algorithm' not in params:
            params['algorithm'] = algorithm

        if 'additional_params' in params:
            additional_params = params['additional_params']

        root_metadata['operation_tag'] = ''

        if class_name == 'SimulationState':
            return

        storage_manager.set_metadata(root_metadata)

        if storage_migrate is False:
            return

        # Create the corresponding datatype to be stored in db
        datatype, generic_attributes = h5.load_with_references(input_file)
        datatype_index = REGISTRY.get_index_for_datatype(datatype.__class__)()
        datatype_index.create_date = root_metadata['create_date']
        datatype_index.disk_size = StorageInterface.compute_size_on_disk(input_file)

        if has_vm is False:
            # Get parent_burst when needed
            time_series_gid = operation_xml_parameters.get('time_series')
            if time_series_gid:
                burst_gid, fk_datatype_group = _set_parent_burst(time_series_gid, root_metadata, storage_manager,
                                                                 input_file)
                if fk_datatype_group is not None:
                    fk_datatype_group = fk_datatype_group + 1
                    datatype_group = dao.get_datatype_by_id(fk_datatype_group)
                    datatype_group.disk_size = datatype_group.disk_size + datatype_index.disk_size
                    datatype_index.fk_datatype_group = fk_datatype_group
                    dao.store_entity(datatype_group)

                    root_metadata['visible'] = "bool:False"
                    generic_attributes.visible = False
                    storage_manager.set_metadata(root_metadata)

                generic_attributes.parent_burst = burst_gid

            alg_json = json.loads(params['algorithm'])
            algorithm = dao.get_algorithm_by_module(alg_json['module'], alg_json['classname'])

            if algorithm is None:
                raise FileMigrationException(alg_json['classname'] + 'data file could not be migrated.')
            operation.algorithm = algorithm
            operation.fk_from_algo = algorithm.id

            operation_data = Operation2ImportData(operation, folder, info_from_xml=operation_xml_parameters)
            operation_entity, _ = import_service.import_operation(operation_data.operation, True)
            possible_burst_id = operation.view_model_gid
            vm = import_service.create_view_model(operation, operation_data, folder,
                                                  generic_attributes, additional_params)

            if 'TimeSeries' in class_name and 'Importer' not in algorithm.classname \
                    and time_series_gid is None:
                burst_config, new_burst = _get_burst_for_migration(possible_burst_id, burst_match_dict,
                                                                   DATE_FORMAT_V4_DB, TvbProfile.current.db.SELECTED_DB)
                if burst_config:
                    root_metadata['parent_burst'] = _parse_gid(burst_config.gid)
                    burst_config.simulator_gid = vm.gid.hex
                    burst_config.fk_simulation = operation.id
                    generic_attributes.parent_burst = burst_config.gid

                    # Creating BurstConfigH5
                    with BurstConfigurationH5(os.path.join(os.path.dirname(input_file),
                                                           'BurstConfiguration_' + burst_config.gid + ".h5")) as f:
                        f.store(burst_config)
                    new_burst_id = dao.store_entity(burst_config).id

                    if new_burst:
                        if burst_config.range1 is None:
                            alg = SimulatorAdapter().view_model_to_has_traits(vm)
                            alg.preconfigure()
                            alg.configure()
                            simulation_history = SimulationHistory()
                            simulation_history.populate_from(alg)
                            history_index = h5.store_complete(simulation_history, op_id,
                                                              operation.project.name,
                                                              generic_attributes=vm.generic_attributes)
                            history_index.fk_from_operation = op_id
                            history_index.fk_parent_burst = burst_config.gid
                            history_index.disk_size = StorageInterface.compute_size_on_disk(
                                os.path.join(folder, 'SimulationHistory_' + history_index.gid) + '.h5')
                            dao.store_entity(history_index)
                        else:

                            ts_operation_group = _migrate_operation_group(burst_config.fk_operation_group,
                                                                          operation_xml_parameters['model'])
                            metric_operation_group = _migrate_operation_group(
                                burst_config.fk_metric_operation_group,
                                operation_xml_parameters['model'])
                            ts_operation_group.name = ts_operation_group.name.replace(class_name, class_name + 'Index')
                            metric_operation_group.name = metric_operation_group.name.replace('DatatypeMeasure',
                                                                                              'DatatypeIndex')
                            dao.store_entity(ts_operation_group)
                            dao.store_entity(metric_operation_group)

                            _migrate_datatype_group(ts_operation_group, burst_config.gid, generic_attributes)
                            _migrate_datatype_group(metric_operation_group, burst_config.gid, generic_attributes)

                            datatype_group = dao.get_datatypegroup_by_op_group_id(burst_config.fk_operation_group)
                            datatype_index.fk_datatype_group = datatype_group.id
                    else:
                        datatype_group = dao.get_datatypegroup_by_op_group_id(burst_config.fk_operation_group)
                        datatype_group.disk_size = dao.get_datatype_group_disk_size(
                            datatype_group.id) + datatype_index.disk_size
                        dao.store_entity(datatype_group)
                        datatype_index.fk_datatype_group = datatype_group.id

                    burst_match_dict[possible_burst_id] = new_burst_id
                    root_metadata['parent_burst'] = _parse_gid(burst_config.gid)
                    storage_manager.set_metadata(root_metadata)

            os.remove(os.path.join(folder, OPERATION_XML))

            # Change range values
            if operation.operation_group is not None:
                for key, value in json.loads(operation.range_values).items():
                    if key.count('_') > 1:
                        first_underscore = key.index('_')
                        last_underscore = key.rfind('_')
                        replacement = key[:first_underscore] + '.' + key[last_underscore + 1:]

                        operation.range_values = operation.range_values.replace(key, replacement)
                    elif isinstance(value, str):
                        replaced_gid = value.replace('-', '')
                        operation.range_values = operation.range_values.replace(value, replaced_gid)
                    # else:
                    #     operation.range_values = operation.range_values.replace(str(value), '\"' + str(value) + '\"')
                dao.store_entity(operation)

        # Populate datatype
        datatype_index.fill_from_has_traits(datatype)
        datatype_index.fill_from_generic_attributes(generic_attributes)
        datatype_index.fk_from_operation = op_id

        # Finally store new datatype in db
        dao.store_entity(datatype_index)
    except Exception as excep:
        if os.path.exists(input_file):
            os.remove(input_file)

        raise FileMigrationException('An unexpected error appeared when migrating file: ' + input_file + '.' + \
                                     ' The exception message: ' + type(excep).__name__ + ': ' + str(excep))
