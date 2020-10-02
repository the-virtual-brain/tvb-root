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
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Upgrade script from H5 version 4 to version 5 (for tvb release 2.0)

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import json
import os
import sys
import uuid
import numpy
from datetime import datetime
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter, CortexViewModel
from tvb.basic.neotraits._attr import Range
from tvb.core.entities.file.simulator.burst_configuration_h5 import BurstConfigurationH5
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistory
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import REGISTRY
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import IncompatibleFileManagerException, MissingDataSetException
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neotraits._h5accessors import DataSetMetaData
from tvb.core.neotraits._h5core import H5File
from tvb.core.neotraits.h5 import STORE_STRING
from tvb.core.services.import_service import OPERATION_XML, ImportService, Operation2ImportData
from tvb.datatypes.sensors import SensorTypes

LOGGER = get_logger(__name__)
FIELD_SURFACE_MAPPING = "has_surface_mapping"
FIELD_VOLUME_MAPPING = "has_volume_mapping"


def _migrate_connectivity(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['number_of_connections'] = int(root_metadata['number_of_connections'])
    root_metadata['number_of_regions'] = int(root_metadata['number_of_regions'])

    if root_metadata['undirected'] == "0":
        root_metadata['undirected'] = "bool:False"
    else:
        root_metadata['undirected'] = "bool:True"

    if root_metadata['saved_selection'] == 'null':
        root_metadata['saved_selection'] = '[]'

    metadata = ['centres', 'region_labels', 'tract_lengths', 'weights']
    extra_metadata = ['orientations', 'areas', 'cortical', 'hemispheres', 'orientations']
    storage_manager = kwargs['storage_manager']

    for mt in extra_metadata:
        try:
            storage_manager.get_metadata(mt)
            metadata.append(mt)
        except MissingDataSetException:
            pass

    if kwargs['operation_xml_parameters']['normalization'] == 'none':
        del kwargs['operation_xml_parameters']['normalization']

    storage_manager.remove_metadata('Mean non zero', 'tract_lengths')
    storage_manager.remove_metadata('Min. non zero', 'tract_lengths')
    storage_manager.remove_metadata('Var. non zero', 'tract_lengths')
    storage_manager.remove_metadata('Mean non zero', 'weights')
    storage_manager.remove_metadata('Min. non zero', 'weights')
    storage_manager.remove_metadata('Var. non zero', 'weights')

    _migrate_dataset_metadata(metadata, storage_manager)

    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_surface(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['edge_max_length'] = float(root_metadata['edge_max_length'])
    root_metadata['edge_mean_length'] = float(root_metadata['edge_mean_length'])
    root_metadata['edge_min_length'] = float(root_metadata['edge_min_length'])
    root_metadata['number_of_split_slices'] = int(root_metadata['number_of_split_slices'])
    root_metadata['number_of_triangles'] = int(root_metadata['number_of_triangles'])
    root_metadata['number_of_vertices'] = int(root_metadata['number_of_vertices'])

    root_metadata['zero_based_triangles'] = "bool:" + root_metadata['zero_based_triangles'][:1].upper() \
                                            + root_metadata['zero_based_triangles'][1:]
    root_metadata['bi_hemispheric'] = "bool:" + root_metadata['bi_hemispheric'][:1].upper() \
                                      + root_metadata['bi_hemispheric'][1:]
    root_metadata['valid_for_simulations'] = "bool:" + root_metadata['valid_for_simulations'][:1].upper() \
                                             + root_metadata['valid_for_simulations'][1:]

    root_metadata["surface_type"] = root_metadata["surface_type"].replace("\"", '')

    operation_xml_parameters = kwargs['operation_xml_parameters']
    if root_metadata['zero_based_triangles'] == 'bool:True':
        operation_xml_parameters['zero_based_triangles'] = True
    else:
        operation_xml_parameters['zero_based_triangles'] = False

    if 'should_center' in operation_xml_parameters:
        if operation_xml_parameters['should_center'] == 'bool:True':
            operation_xml_parameters['should_center'] = True
        else:
            operation_xml_parameters['should_center'] = False

    kwargs['storage_manager'].store_data('split_triangles', [])

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

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])

    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_sensors(datasets, **kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['number_of_sensors'] = int(root_metadata['number_of_sensors'])
    root_metadata['sensors_type'] = root_metadata["sensors_type"].replace("\"", '')
    root_metadata['has_orientation'] = "bool:" + root_metadata['has_orientation'][:1].upper() \
                                       + root_metadata['has_orientation'][1:]

    storage_manager = kwargs['storage_manager']
    storage_manager.remove_metadata('Size', 'labels')
    storage_manager.remove_metadata('Size', 'locations')
    storage_manager.remove_metadata('Variance', 'locations')
    storage_manager = _bytes_ds_to_string_ds(storage_manager, 'labels')
    _migrate_dataset_metadata(datasets, storage_manager)
    algorithm = '{"classname": "SensorsImporter", "module": "tvb.adapters.uploaders.sensors_importer"}'
    kwargs['operation_xml_parameters']['sensors_file'] = kwargs['input_file']
    return algorithm, kwargs['operation_xml_parameters']


def _migrate_eeg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypes.TYPE_EEG.value
    return {'algorithm': algorithm, 'operation_xml_parameters': operation_xml_parameters}


def _migrate_meg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations', 'orientations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypes.TYPE_MEG.value
    return {'algorithm': algorithm, 'operation_xml_parameters': operation_xml_parameters}


def _migrate_seeg_sensors(**kwargs):
    algorithm, operation_xml_parameters = _migrate_sensors(['labels', 'locations'], **kwargs)
    operation_xml_parameters['sensors_type'] = SensorTypes.TYPE_INTERNAL.value
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
    matrix_metadata['Shape'] = str(matrix_metadata['Shape'], 'utf-8')
    matrix_metadata['dtype'] = str(matrix_metadata['dtype'], 'utf-8')
    matrix_metadata['format'] = str(matrix_metadata['format'], 'utf-8')
    storage_manager.set_metadata(matrix_metadata, 'matrix')

    equation = json.loads(root_metadata['equation'])
    equation['type'] = equation['__mapped_class']
    del equation['__mapped_class']
    del equation['__mapped_module']

    root_metadata['equation'] = json.dumps(equation)

    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['surface'] = root_metadata['surface']
    operation_xml_parameters['cutoff'] = float(operation_xml_parameters['cutoff'])
    equation_name = operation_xml_parameters['equation']
    equation = getattr(sys.modules['tvb.datatypes.equations'],
                       equation_name)()
    operation_xml_parameters['equation'] = equation
    parameters = {}

    for xml_param in operation_xml_parameters:
        if '_parameters_parameters_' in xml_param:
            new_key = xml_param.replace('_parameters_option_' + equation_name + '_parameters_parameters_',
                                        '.parameters.')
            param_name_start_idx = new_key.rfind('.')
            param_name = new_key[param_name_start_idx + 1:]
            parameters[param_name] = float(operation_xml_parameters[xml_param])
    additional_params = [['equation', 'parameters', parameters]]
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_connectivity_annotations(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])
    root_metadata['written_by'] = "tvb.adapters.datatypes.h5.annotation_h5.ConnectivityAnnotationsH5"

    _migrate_dataset_metadata(['region_annotations'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_time_series(metadata, **kwargs):
    root_metadata = kwargs['root_metadata']
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters.pop('', None)
    if 'surface' in operation_xml_parameters and operation_xml_parameters['surface'] == '':
        operation_xml_parameters['surface'] = None
    if 'stimulus' in operation_xml_parameters and operation_xml_parameters['stimulus'] == '':
        operation_xml_parameters['stimulus'] = None

    root_metadata.pop(FIELD_SURFACE_MAPPING)
    root_metadata.pop(FIELD_VOLUME_MAPPING)
    _pop_lengths(kwargs['root_metadata'])

    root_metadata['sample_period'] = float(root_metadata['sample_period'])
    root_metadata['start_time'] = float(root_metadata['start_time'])

    root_metadata["sample_period_unit"] = root_metadata["sample_period_unit"].replace("\"", '')
    root_metadata[DataTypeMetaData.KEY_TITLE] = root_metadata[DataTypeMetaData.KEY_TITLE].replace("\"", '')
    _migrate_dataset_metadata(metadata, kwargs['storage_manager'])

    if len(operation_xml_parameters) == 0:
        return operation_xml_parameters, []

    coupling = getattr(sys.modules['tvb.simulator.coupling'], operation_xml_parameters['coupling'])()
    operation_xml_parameters['coupling'] = coupling

    model_name = operation_xml_parameters['model']
    model = getattr(sys.modules['tvb.simulator.models'], model_name)()
    operation_xml_parameters['model'] = model

    integrator_name = operation_xml_parameters['integrator']
    integrator = getattr(sys.modules['tvb.core.entities.file.simulator.view_model'],
                         integrator_name + 'ViewModel')()
    operation_xml_parameters['integrator'] = integrator

    monitors = operation_xml_parameters['monitors']
    for i in range(len(monitors)):
        monitor = getattr(sys.modules['tvb.core.entities.file.simulator.view_model'],
                          operation_xml_parameters['monitors'][i] + 'ViewModel')()
        operation_xml_parameters['monitors'][i] = monitor

    operation_xml_parameters['conduction_speed'] = float(operation_xml_parameters['simulation_length'])
    operation_xml_parameters['simulation_length'] = float(operation_xml_parameters['simulation_length'])

    additional_params = []
    for xml_param in operation_xml_parameters:
        if 'model_parameters_option_' in xml_param and 'range_parameters' not in xml_param:
            new_param = xml_param.replace('model_parameters_option_' + model_name + '_', '')
            list_param = operation_xml_parameters[xml_param]

            if model_name + '_variables_of_interest' not in xml_param:
                try:
                    list_param = numpy.asarray(eval(list_param.replace(' ', ', ')))
                except AttributeError:
                    list_param = numpy.asarray(list_param)

            additional_params.append(['model', new_param, list_param])
        elif 'integrator_parameters_option_' in xml_param:
            new_param = xml_param.replace('integrator_parameters_option_' + integrator_name + '_', '')
            additional_params.append(['integrator', new_param, float(operation_xml_parameters[xml_param])])
    return operation_xml_parameters, additional_params


def _migrate_time_series_region(**kwargs):
    operation_xml_parameters, additional_params = _migrate_time_series(['data', 'time'], **kwargs)
    root_metadata = kwargs['root_metadata']
    root_metadata['region_mapping'] = _parse_gid(root_metadata['region_mapping'])
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_time_series_surface(**kwargs):
    operation_xml_parameters, additional_params = _migrate_time_series(['data', 'time'], **kwargs)
    surface_gid = operation_xml_parameters['surface']
    cortical_surface = CortexViewModel()
    cortical_surface.surface_gid = uuid.UUID(surface_gid)
    cortical_surface.region_mapping_data = uuid.UUID(operation_xml_parameters['surface_parameters_region_mapping_data'])
    cortical_surface.local_connectivity = uuid.UUID(operation_xml_parameters['surface_parameters_local_connectivity'])
    cortical_surface.coupling_strength = numpy.asarray(eval(operation_xml_parameters[
                                                                'surface_parameters_coupling_strength'].replace(' ',
                                                                                                                ', ')))
    operation_xml_parameters['surface'] = cortical_surface
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _set_sensors_view_model_attributes(operation_xml_parameters, sensors_type):
    setattr(operation_xml_parameters['monitors'][0], 'region_mapping',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_region_mapping'])
    setattr(operation_xml_parameters['monitors'][0], 'sensors',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_sensors'])
    setattr(operation_xml_parameters['monitors'][0], 'projection',
            operation_xml_parameters['monitors_parameters_option_' + sensors_type + '_projection'])


def _migrate_time_series_sensors(**kwargs):
    sensors_type = kwargs['operation_xml_parameters']['monitors'][0]
    operation_xml_parameters, additional_params = _migrate_time_series(['data', 'time'], **kwargs)
    root_metadata = kwargs['root_metadata']
    root_metadata['sensors'] = _parse_gid(root_metadata['sensors'])

    _set_sensors_view_model_attributes(operation_xml_parameters, sensors_type)
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_time_series_volume(**kwargs):
    operation_xml_parameters, additional_params = _migrate_time_series(['data'], **kwargs)
    root_metadata = kwargs['root_metadata']
    root_metadata['nr_dimensions'] = int(root_metadata['nr_dimensions'])
    root_metadata['sample_rate'] = float(root_metadata['sample_rate'])
    root_metadata['volume'] = _parse_gid(root_metadata['volume'])
    root_metadata['volume'] = root_metadata['volume']
    root_metadata.pop('nr_dimensions')
    root_metadata.pop('sample_rate')
    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_volume(**kwargs):
    root_metadata = kwargs['root_metadata']
    kwargs['root_metadata']['voxel_unit'] = root_metadata['voxel_unit'].replace("\"", '')

    operation_xml_parameters = kwargs['operation_xml_parameters']
    if operation_xml_parameters['connectivity'] == '':
        operation_xml_parameters['connectivity'] = None

    if operation_xml_parameters['apply_corrections'] == 'bool:True':
        operation_xml_parameters['apply_corrections'] = True
    else:
        operation_xml_parameters['apply_corrections'] = False

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
    _migrate_dataset_metadata(['amplitude', 'array_data', 'frequencies', 'phase', 'power'], kwargs['storage_manager'])
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
    storage_manager.store_data('array_data', numpy.asarray(array_data, dtype=numpy.float64))
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

    if 'parent_burst' in root_metadata:
        return {'operation_xml_parameters': operation_xml_parameters}

    # If we encounter an FCD we expect it to have more FCDs and ConnectivityMeasure's
    # so we can only set parent_burst attribute on all files
    folder_name = os.path.dirname(kwargs['input_file'])
    for file_name in os.listdir(folder_name):
        if file_name != OPERATION_XML and file_name not in kwargs['input_file']:
            storage_manager = HDF5StorageManager(folder_name, file_name)
            root_metadata = storage_manager.get_metadata()
            _set_parent_burst(operation_xml_parameters['time_series'], root_metadata, storage_manager, True)

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

    if operation_xml_parameters['detrend'] == 'on':
        operation_xml_parameters['detrend'] = True
    else:
        operation_xml_parameters['detrend'] = False
    operation_xml_parameters['segment_length'] = root_metadata['segment_length']

    _migrate_dataset_metadata(['amplitude', 'array_data', 'average_power',
                               'normalised_average_power', 'phase', 'power'], kwargs['storage_manager'])
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

    _migrate_dataset_metadata(['component_time_series', 'fractions',
                               'norm_source', 'normalised_component_time_series',
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

    _migrate_dataset_metadata(['array_data'], kwargs['storage_manager'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_datatype_measure(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['written_by'] = 'tvb.core.entities.file.simulator.datatype_measure_h5.DatatypeMeasureH5'
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


def _migrate_stimuli(datasets, root_metadata, storage_manager):
    _migrate_one_stimuli_param(root_metadata, 'spatial')
    _migrate_one_stimuli_param(root_metadata, 'temporal')

    for dataset in datasets:
        weights = json.loads(root_metadata[dataset])
        storage_manager.store_data(dataset, weights)
        _migrate_dataset_metadata([dataset], storage_manager)
        root_metadata.pop(dataset)


def _migrate_stimuli_region(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['connectivity'] = _parse_gid(root_metadata['connectivity'])
    operation_xml_parameters = kwargs['operation_xml_parameters']

    operation_xml_parameters['weight'] = numpy.asarray(json.loads(root_metadata['weight']), dtype=numpy.float64)
    _migrate_stimuli(['weight'], root_metadata, kwargs['storage_manager'])
    additional_params = [_migrate_stimuli_equation_params(operation_xml_parameters, 'temporal')]

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_stimuli_surface(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['surface'] = _parse_gid(root_metadata['surface'])
    operation_xml_parameters = kwargs['operation_xml_parameters']
    operation_xml_parameters['focal_points_triangles'] = numpy.asarray(
        json.loads(root_metadata['focal_points_triangles']),
        dtype=numpy.int)
    _migrate_stimuli(['focal_points_surface', 'focal_points_triangles'], root_metadata, kwargs['storage_manager'])
    additional_params = [_migrate_stimuli_equation_params(operation_xml_parameters, 'temporal'),
                         _migrate_stimuli_equation_params(operation_xml_parameters, 'spatial')]

    return {'operation_xml_parameters': operation_xml_parameters, 'additional_params': additional_params}


def _migrate_value_wrapper(**kwargs):
    root_metadata = kwargs['root_metadata']
    root_metadata['data_type'] = root_metadata['data_type'].replace("\"", '')
    root_metadata['written_by'] = "tvb.adapters.datatypes.h5.mapped_value_h5.ValueWrapperH5"
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


def _migrate_simulation_state(**kwargs):
    os.remove(kwargs['input_file'])
    return {'operation_xml_parameters': kwargs['operation_xml_parameters']}


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


def _bytes_ds_to_string_ds(storage_manager, ds_name):
    bytes_labels = storage_manager.get_data(ds_name)
    string_labels = []
    for i in range(len(bytes_labels)):
        string_labels.append(str(bytes_labels[i], 'utf-8'))

    storage_manager.remove_data(ds_name)
    storage_manager.store_data(ds_name, numpy.asarray(string_labels).astype(STORE_STRING))
    return storage_manager


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


def _create_new_burst(burst_params, root_metadata):
    burst_config = BurstConfiguration(burst_params['fk_project'])
    burst_config.datatypes_number = burst_params['datatypes_number']
    burst_config.dynamic_ids = burst_params['dynamic_ids']
    burst_config.error_message = burst_params['error_message']
    burst_config.finish_time = datetime.strptime(burst_params['finish_time'], '%Y-%m-%d %H:%M:%S.%f')
    burst_config.fk_metric_operation_group = burst_params['fk_metric_operation_group']
    burst_config.fk_operation_group = burst_params['fk_operation_group']
    burst_config.fk_project = burst_params['fk_project']
    burst_config.fk_simulation = burst_params['fk_simulation']
    burst_config.name = burst_params['name']
    burst_config.range_1 = burst_params['range_1']
    burst_config.range_2 = burst_params['range_2']
    burst_config.start_time = datetime.strptime(burst_params['start_time'], '%Y-%m-%d %H:%M:%S.%f')
    burst_config.status = burst_params['status']
    root_metadata['parent_burst'] = _parse_gid(burst_config.gid)

    return burst_config


def _set_parent_burst(time_series_gid, root_metadata, storage_manager, is_ascii=False):
    ts = dao.get_datatype_by_gid(
        time_series_gid.replace('-', '').replace('urn:uuid:', ''))
    burst_gid = _parse_gid(ts.fk_parent_burst)
    if is_ascii:
        burst_gid = burst_gid.encode('ascii', 'ignore')
    root_metadata['parent_burst'] = burst_gid
    storage_manager.set_metadata(root_metadata)
    return ts.fk_parent_burst


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
    'SimulationState': _migrate_simulation_state
}


def update(input_file):
    """
    :param input_file: the file that needs to be converted to a newer file storage version.
    """

    # The first step is to check based on the path if this function call is part of the migration of the whole storage
    # folder or just one datatype (in the first case the second to last element in the path is a number
    split_path = input_file.split(os.path.sep)
    storage_migrate = True
    try:
        # Change file names only for storage migration
        op_id = int(split_path[-2])

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

    if not os.path.isfile(input_file):
        raise IncompatibleFileManagerException("Not yet implemented update for file %s" % input_file)

    # Obtain storage manager and metadata
    folder, file_name = os.path.split(input_file)
    storage_manager = HDF5StorageManager(folder, file_name)
    root_metadata = storage_manager.get_metadata()

    if DataTypeMetaData.KEY_CLASS_NAME not in root_metadata:
        raise IncompatibleFileManagerException("File %s received for upgrading 4 -> 5 is not valid, due to missing "
                                               "metadata: %s" % (input_file, DataTypeMetaData.KEY_CLASS_NAME))

    # In the new format all metadata has the 'TVB_%' format, where '%' starts with a lowercase letter
    lowercase_keys = []
    for key, value in root_metadata.items():
        root_metadata[key] = str(value, 'utf-8')
        lowercase_keys.append(_lowercase_first_character(key))
        storage_manager.remove_metadata(key)

    # Update DATA_VERSION
    root_metadata = dict(zip(lowercase_keys, list(root_metadata.values())))
    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION
    class_name = root_metadata["type"]

    # UPDATE CREATION DATE
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace('datetime:', '')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(':', '-')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(' ', ',')

    # OBTAIN THE MODULE (for a few data types the old module doesn't exist anymore, in those cases the attr
    # will be set later
    try:
        datatype_class = getattr(sys.modules[root_metadata["module"]],
                                 root_metadata["type"])
        h5_class = REGISTRY.get_h5file_for_datatype(datatype_class)
        root_metadata[H5File.KEY_WRITTEN_BY] = h5_class().get_class_path()
    except KeyError:
        pass

    # Other general modifications
    root_metadata['gid'] = _parse_gid(root_metadata['gid'])
    root_metadata.pop("type")
    root_metadata.pop("module")
    root_metadata.pop('data_version')

    files_in_folder = os.listdir(folder)
    import_service = ImportService()
    algorithm = ''

    operation_xml_parameters = {}  # params that are needed for the view_model but are not in the Operation.xml file
    additional_params = None  # params that can't be jsonified with json.dumps
    has_vm = False
    operation = None
    # Take information out from the Operation.xml file
    if OPERATION_XML in files_in_folder:
        operation_file_path = os.path.join(folder, OPERATION_XML)
        project = dao.get_project_by_name(split_path[-3])
        xml_operation, operation_xml_parameters, algorithm = \
            import_service.build_operation_from_file(project, operation_file_path)
        operation = dao.get_operation_by_id(op_id)
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
    storage_manager.set_metadata(root_metadata)

    if storage_migrate is False or class_name in ['SimulationState', 'DatatypeMeasure']:
        return

    # Create the corresponding datatype to be stored in db
    datatype, generic_attributes = h5.load_with_references(input_file)
    datatype_index = REGISTRY.get_index_for_datatype(datatype.__class__)()
    datatype_index.create_date = root_metadata['create_date']

    if has_vm is False:
        # Get parent_burst when needed
        if 'time_series' in operation_xml_parameters:
            burst_gid = _set_parent_burst(operation_xml_parameters['time_series'], root_metadata, storage_manager)
            generic_attributes.parent_burst = burst_gid

        alg_json = json.loads(params['algorithm'])
        algorithm = dao.get_algorithm_by_module(alg_json['module'], alg_json['classname'])
        operation.algorithm = algorithm
        operation.fk_from_algo = algorithm.id

        operation_data = Operation2ImportData(operation, folder, info_from_xml=operation_xml_parameters)
        operation_entity, _ = import_service.import_operation(operation_data.operation)
        possible_burst_id = operation.view_model_gid
        vm = import_service.create_view_model(operation, operation_data, folder, additional_params)
        vm.generic_attributes = generic_attributes

        if 'TimeSeries' in class_name:
            alg = SimulatorAdapter().view_model_to_has_traits(vm)
            alg.preconfigure()
            alg.configure()
            simulation_history = SimulationHistory()
            simulation_history.populate_from(alg)
            history_index = h5.store_complete(simulation_history, folder, generic_attributes=vm.generic_attributes)
            history_index.fk_from_operation = op_id
            history_index.fk_parent_burst = possible_burst_id

            burst_params = dao.get_burst_for_migration(possible_burst_id)
            if burst_params is not None:
                burst_config = _create_new_burst(burst_params, root_metadata)
                burst_config.simulator_gid = vm.gid.hex
                burst_config.fk_simulation = operation.id
                generic_attributes.parent_burst = burst_config.gid

                # Creating BurstConfigH5
                with BurstConfigurationH5(os.path.join(os.path.dirname(input_file), 'BurstConfiguration_'
                                                                                    + burst_config.gid + ".h5")) as f:
                    f.store(burst_config)

                history_index.fk_parent_burst = burst_config.gid
                dao.store_entity(history_index)
                dao.store_entity(burst_config)
                root_metadata['parent_burst'] = _parse_gid(burst_config.gid)
                storage_manager.set_metadata(root_metadata)

        os.remove(os.path.join(folder, OPERATION_XML))

    # Populate datatype
    datatype_index.fill_from_has_traits(datatype)
    datatype_index.fill_from_generic_attributes(generic_attributes)
    datatype_index.fk_from_operation = op_id

    # Finally store new datatype in db
    dao.store_entity(datatype_index)
