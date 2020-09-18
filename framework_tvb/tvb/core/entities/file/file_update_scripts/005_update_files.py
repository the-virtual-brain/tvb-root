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
import numpy
from tvb.adapters.analyzers.bct_adapters import BaseBCTModel
from tvb.adapters.analyzers.fcd_adapter import FCDAdapterModel
from tvb.adapters.analyzers.ica_adapter import ICAAdapterModel
from tvb.adapters.analyzers.metrics_group_timeseries import TimeseriesMetricsAdapterModel
from tvb.adapters.analyzers.node_coherence_adapter import NodeCoherenceModel
from tvb.adapters.analyzers.node_complex_coherence_adapter import NodeComplexCoherenceModel
from tvb.adapters.analyzers.node_covariance_adapter import NodeCovarianceAdapterModel
from tvb.adapters.analyzers.pca_adapter import PCAAdapterModel
from tvb.adapters.creators.local_connectivity_creator import LocalConnectivityCreatorModel
from tvb.adapters.creators.stimulus_creator import RegionStimulusCreatorModel, SurfaceStimulusCreatorModel
from tvb.adapters.datatypes.h5.annotation_h5 import ConnectivityAnnotationsH5
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapperH5
from tvb.adapters.uploaders.brco_importer import BRCOImporterModel
from tvb.adapters.uploaders.connectivity_measure_importer import ConnectivityMeasureImporterModel
from tvb.adapters.uploaders.nifti_importer import NIFTIImporterModel
from tvb.adapters.uploaders.projection_matrix_importer import ProjectionMatrixImporterModel
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporterModel
from tvb.adapters.uploaders.sensors_importer import SensorsImporterModel
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporterModel
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporterModel
from tvb.adapters.visualizers.cross_correlation import CrossCorrelationVisualizerModel
from tvb.adapters.visualizers.fourier_spectrum import FourierSpectrumModel
from tvb.adapters.visualizers.wavelet_spectrogram import WaveletSpectrogramVisualizerModel
from tvb.basic.neotraits.ex import TraitTypeError, TraitAttributeError
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
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


LOGGER = get_logger(__name__)
FIELD_SURFACE_MAPPING = "has_surface_mapping"
FIELD_VOLUME_MAPPING = "has_volume_mapping"
GID_PREFIX = "urn:uuid:"


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
        conn_metadata = DataSetMetaData.from_array(storage_manager.get_data(dataset)).to_dict()
        storage_manager.set_metadata(conn_metadata, dataset)
        metadata = storage_manager.get_metadata(dataset)
        if 'Variance' in metadata:
            storage_manager.remove_metadata('Variance', dataset)
        if 'Size' in metadata:
            storage_manager.remove_metadata('Size',dataset)


def _migrate_one_stimuli_param(root_metadata, param_name):
    param = json.loads(root_metadata[param_name])
    new_param = dict()
    new_param['type'] = param['__mapped_class']
    new_param['parameters'] = param['parameters']
    root_metadata[param_name] = json.dumps(new_param)


def _migrate_stimuli(root_metadata, storage_manager, datasets):
    _migrate_one_stimuli_param(root_metadata, 'spatial')
    _migrate_one_stimuli_param(root_metadata, 'temporal')

    for dataset in datasets:
        weights = eval(root_metadata[dataset])
        storage_manager.store_data(dataset, weights)
        _migrate_dataset_metadata([dataset], storage_manager)
        root_metadata.pop(dataset)


def _migrate_time_series(root_metadata, storage_manager, class_name, dependent_attributes):
    root_metadata.pop(FIELD_SURFACE_MAPPING)
    root_metadata.pop(FIELD_VOLUME_MAPPING)
    _pop_lengths(root_metadata)
    metadata = ['data', 'time']

    if class_name != 'TimeSeriesVolume':
        root_metadata['nr_dimensions'] = int(root_metadata['nr_dimensions'])
        root_metadata['sample_rate'] = float(root_metadata['sample_rate'])
        view_model_class = SimulatorAdapterModel
    else:
        metadata.pop(1)
        view_model_class = NIFTIImporterModel

    root_metadata['sample_period'] = float(root_metadata['sample_period'])
    root_metadata['start_time'] = float(root_metadata['start_time'])

    root_metadata["sample_period_unit"] = root_metadata["sample_period_unit"].replace("\"", '')
    root_metadata[DataTypeMetaData.KEY_TITLE] = root_metadata[DataTypeMetaData.KEY_TITLE].replace("\"", '')
    _migrate_dataset_metadata(metadata, storage_manager)

    if class_name == 'TimeSeriesRegion':
        root_metadata['region_mapping'] = GID_PREFIX + root_metadata['region_mapping']
        root_metadata['connectivity'] = GID_PREFIX + root_metadata['connectivity']

        dependent_attributes['connectivity'] = root_metadata['connectivity']
        dependent_attributes['region_mapping'] = root_metadata['region_mapping']
    elif class_name == 'TimeSeriesSurface':
        root_metadata['surface'] = GID_PREFIX + root_metadata['surface']
        dependent_attributes['surface'] = root_metadata['surface']
    elif class_name in ['TimeSeriesEEG', 'TimeSeriesMEG', 'TimeSeriesSEEG']:
        root_metadata['sensors'] = GID_PREFIX + root_metadata['sensors']
        dependent_attributes['sensors'] = root_metadata['sensors']
    else:
        root_metadata['volume'] = GID_PREFIX + root_metadata['volume']
        dependent_attributes['volume'] = root_metadata['volume']
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('sample_rate')

    return dependent_attributes, view_model_class


def _create_new_burst(project_id, root_metadata):
    burst_config = BurstConfiguration(project_id)
    burst_config.name = 'simulation_' + str(dao.get_max_burst_id() + 1)
    dao.store_entity(burst_config)
    root_metadata['parent_burst'] = GID_PREFIX + burst_config.gid
    return burst_config.gid


def update(input_file):
    """
    :param input_file: the file that needs to be converted to a newer file storage version.
    """

    # The first step is to check based on the path if this function call is part of the migration of the whole storage
    # folder or just one datatype (in the first case the second to last element in the path is a number
    split_path = input_file.split('\\')
    storage_migrate = True
    try:
        # Change file names only for storage migration
        op_id = int(split_path[-2])
        replaced_input_file = input_file.replace('-', '')
        replaced_input_file = replaced_input_file.replace('BrainSkull', 'Surface')
        replaced_input_file = replaced_input_file.replace('CorticalSurface', 'Surface')
        replaced_input_file = replaced_input_file.replace('SkinAir', 'Surface')
        replaced_input_file = replaced_input_file.replace('BrainSkull', 'Surface')
        replaced_input_file = replaced_input_file.replace('SkullSkin', 'Surface')
        replaced_input_file = replaced_input_file.replace('EEGCap', 'Surface')
        replaced_input_file = replaced_input_file.replace('FaceSurface', 'Surface')
        os.rename(input_file, replaced_input_file)
        input_file = replaced_input_file
    except ValueError:
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

    # OBTAIN THE MODULE (for a few datatypes the old module doesn't exist anymore, in those cases the attr
    # will be set later
    try:
        datatype_class = getattr(sys.modules[root_metadata["module"]],
                             root_metadata["type"])
        h5_class = REGISTRY.get_h5file_for_datatype(datatype_class)
        root_metadata[H5File.KEY_WRITTEN_BY] = h5_class.__module__ + '.' + h5_class.__name__
    except KeyError:
        pass

    # Other general modifications
    root_metadata['user_tag_1'] = ''
    root_metadata['gid'] = GID_PREFIX + root_metadata['gid']

    root_metadata.pop("type")
    root_metadata.pop("module")
    root_metadata.pop('data_version')

    dependent_attributes = {} # where is the case the datatype will have a dict of the datatype it depends on
    changed_values = {} # where there are changes between the view model class attributes and the OPERATIONS.parameters
    # attr in the old DB, those changed will be marked in this dict
    view_model_class = None

    # HERE STARTS THE PART WHICH IS SPECIFIC TO EACH H5 FILE #

    if class_name == 'Connectivity':
        root_metadata['number_of_connections'] = int(root_metadata['number_of_connections'])
        root_metadata['number_of_regions'] = int(root_metadata['number_of_regions'])

        if root_metadata['undirected'] == "0":
            root_metadata['undirected'] = "bool:False"
        else:
            root_metadata['undirected'] = "bool:True"

        if root_metadata['saved_selection'] == 'null':
            root_metadata['saved_selection'] = '[]'

        metadata = ['areas', 'centres', 'orientations', 'region_labels', 'tract_lengths', 'weights']
        extra_metadata = ['cortical', 'hemispheres']

        for mt in extra_metadata:
            try:
                storage_manager.get_metadata(mt)
                metadata.append(mt)
            except MissingDataSetException:
                pass

        storage_manager.remove_metadata('Mean non zero', 'tract_lengths')
        storage_manager.remove_metadata('Min. non zero', 'tract_lengths')
        storage_manager.remove_metadata('Var. non zero', 'tract_lengths')
        storage_manager.remove_metadata('Mean non zero', 'weights')
        storage_manager.remove_metadata('Min. non zero', 'weights')
        storage_manager.remove_metadata('Var. non zero', 'weights')

        _migrate_dataset_metadata(metadata, storage_manager)
        view_model_class = ZIPConnectivityImporterModel

    elif class_name in ['BrainSkull', 'CorticalSurface', 'SkinAir', 'BrainSkull', 'SkullSkin', 'EEGCap', 'FaceSurface']:
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

        if root_metadata['zero_based_triangles'] == 'bool:True':
            changed_values['zero_based_triangles'] = True
        else:
            changed_values['zero_based_triangles'] = False

        storage_manager.store_data('split_triangles', [])

        _migrate_dataset_metadata(['split_triangles', 'triangle_normals', 'triangles', 'vertex_normals', 'vertices'],
                                  storage_manager)
        view_model_class = ZIPSurfaceImporterModel

    elif class_name == 'RegionMapping':
        _pop_lengths(root_metadata)
        _pop_common_metadata(root_metadata)

        root_metadata['surface'] = GID_PREFIX + root_metadata['surface']
        root_metadata['connectivity'] = GID_PREFIX + root_metadata['connectivity']

        dependent_attributes['connectivity'] = root_metadata['connectivity']
        dependent_attributes['surface'] = root_metadata['surface']

        _migrate_dataset_metadata(['array_data'], storage_manager)
        view_model_class = RegionMappingImporterModel

    elif 'Sensors' in class_name:
        root_metadata['number_of_sensors'] = int(root_metadata['number_of_sensors'])
        root_metadata['sensors_type'] = root_metadata["sensors_type"].replace("\"", '')
        root_metadata['has_orientation'] = "bool:" + root_metadata['has_orientation'][:1].upper() \
                                           + root_metadata['has_orientation'][1:]

        storage_manager.remove_metadata('Size', 'labels')
        storage_manager.remove_metadata('Size', 'locations')
        storage_manager.remove_metadata('Variance', 'locations')
        storage_manager = _bytes_ds_to_string_ds(storage_manager, 'labels')

        datasets = ['labels', 'locations']

        if 'MEG' in class_name:
            storage_manager.remove_metadata('Size', 'orientations')
            storage_manager.remove_metadata('Variance', 'orientations')
            datasets.append('orientations')

        _migrate_dataset_metadata(datasets, storage_manager)
        view_model_class = SensorsImporterModel

    elif 'Projection' in class_name:
        root_metadata['sensors'] = GID_PREFIX + root_metadata['sensors']
        root_metadata['sources'] = GID_PREFIX + root_metadata['sources']
        root_metadata['written_by'] = "tvb.adapters.datatypes.h5.projections_h5.ProjectionMatrixH5"
        root_metadata['projection_type'] = root_metadata["projection_type"].replace("\"", '')

        storage_manager.remove_metadata('Size', 'projection_data')
        storage_manager.remove_metadata('Variance', 'projection_data')

        changed_values['surface'] = root_metadata['surface']

        _migrate_dataset_metadata(['projection_data'], storage_manager)
        view_model_class = ProjectionMatrixImporterModel

    elif class_name == 'LocalConnectivity':
        root_metadata['cutoff'] = float(root_metadata['cutoff'])
        root_metadata['surface'] = GID_PREFIX + root_metadata['surface']

        storage_manager.remove_metadata('shape', 'matrix')

        matrix_metadata = storage_manager.get_metadata('matrix')
        matrix_metadata['Shape'] = str(matrix_metadata['Shape'], 'utf-8')
        matrix_metadata['dtype'] = str(matrix_metadata['dtype'], 'utf-8')
        matrix_metadata['format'] = str(matrix_metadata['format'], 'utf-8')
        storage_manager.set_metadata(matrix_metadata, 'matrix')

        view_model_class = LocalConnectivityCreatorModel
        dependent_attributes['surface'] = root_metadata['surface']

    elif 'TimeSeries' in class_name:
        dependent_attributes, view_model_class = _migrate_time_series(root_metadata, storage_manager,
                                                                      class_name, dependent_attributes)
    elif 'Volume' in class_name:
        root_metadata['voxel_unit'] = root_metadata['voxel_unit'].replace("\"", '')
        _migrate_dataset_metadata(['origin', 'voxel_size'], storage_manager)

    elif class_name == 'StructuralMRI':
        _pop_lengths(root_metadata)
        _pop_common_metadata(root_metadata)
        root_metadata['volume'] = GID_PREFIX + root_metadata['volume']
        _migrate_dataset_metadata(['array_data'], storage_manager)

    elif class_name == 'CoherenceSpectrum':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['nfft'] = int(root_metadata['nfft'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        array_data = storage_manager.get_data('array_data')
        storage_manager.remove_data('array_data')
        storage_manager.store_data('array_data', numpy.asarray(array_data, dtype=numpy.float64))
        dependent_attributes['source'] = root_metadata['source']

        _migrate_dataset_metadata(['array_data', 'frequency'], storage_manager)
        view_model_class = NodeCoherenceModel

    elif class_name == 'ComplexCoherenceSpectrum':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['epoch_length'] = float(root_metadata['epoch_length'])
        root_metadata['segment_length'] = float(root_metadata['segment_length'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        root_metadata['windowing_function'] = root_metadata['windowing_function'].replace("\"", '')
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        dependent_attributes['source'] = root_metadata['source']
        _migrate_dataset_metadata(['array_data', 'cross_spectrum'], storage_manager)
        view_model_class = NodeComplexCoherenceModel

    elif class_name == 'WaveletCoefficients':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['q_ratio'] = float(root_metadata['q_ratio'])
        root_metadata['sample_period'] = float(root_metadata['sample_period'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        root_metadata['mother'] = root_metadata['mother'].replace("\"", '')
        root_metadata['normalisation'] = root_metadata['normalisation'].replace("\"", '')

        dependent_attributes['source'] = root_metadata['source']
        _migrate_dataset_metadata(['amplitude', 'array_data', 'frequencies', 'phase', 'power'], storage_manager)
        view_model_class = WaveletSpectrogramVisualizerModel

    elif class_name == 'CrossCorrelation':
        changed_values['datatype'] = root_metadata['source']
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        dependent_attributes['source'] = root_metadata['source']
        _migrate_dataset_metadata(['array_data', 'time'], storage_manager)
        view_model_class = CrossCorrelationVisualizerModel

    elif class_name == 'Fcd':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['sp'] = float(root_metadata['sp'])
        root_metadata['sw'] = float(root_metadata['sw'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        dependent_attributes['source'] = root_metadata['source']
        view_model_class = FCDAdapterModel

    elif class_name == 'ConnectivityMeasure':
        _pop_common_metadata(root_metadata)
        _pop_lengths(root_metadata)

        root_metadata['source'] = GID_PREFIX + root_metadata['source']
        root_metadata['connectivity'] = GID_PREFIX + root_metadata['connectivity']

        dependent_attributes['source'] = root_metadata['source']
        _migrate_dataset_metadata(['array_data'], storage_manager)
        view_model_class = ConnectivityMeasureImporterModel

    elif class_name == 'FourierSpectrum':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['segment_length'] = float(root_metadata['segment_length'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        dependent_attributes['source'] = root_metadata['source']
        _migrate_dataset_metadata(['amplitude', 'array_data', 'average_power',
                                   'normalised_average_power', 'phase', 'power'], storage_manager)
        view_model_class = FourierSpectrumModel

    elif class_name == 'IndependentComponents':
        root_metadata['n_components'] = int(root_metadata['n_components'])
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        _migrate_dataset_metadata(['component_time_series', 'mixing_matrix', 'norm_source',
                                   'normalised_component_time_series', 'prewhitening_matrix',
                                   'unmixing_matrix'], storage_manager)
        view_model_class = ICAAdapterModel

    elif class_name == 'CorrelationCoefficients':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        _migrate_dataset_metadata(['array_data'], storage_manager)
        view_model_class = CrossCorrelationVisualizerModel

    elif class_name == 'PrincipalComponents':
        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        _migrate_dataset_metadata(['component_time_series', 'fractions',
                                   'norm_source', 'normalised_component_time_series',
                                   'weights'], storage_manager)
        view_model_class = PCAAdapterModel

    elif class_name == 'Covariance':
        _pop_common_metadata(root_metadata)
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['source'] = GID_PREFIX + root_metadata['source']

        _migrate_dataset_metadata(['array_data'], storage_manager)
        view_model_class = NodeCovarianceAdapterModel
        dependent_attributes['source'] = root_metadata['source']

    elif class_name == 'DatatypeMeasure':
        root_metadata['written_by'] = 'tvb.core.entities.file.simulator.datatype_measure_h5.DatatypeMeasureH5'
        view_model_class = TimeseriesMetricsAdapterModel
        dependent_attributes['source'] = root_metadata['source']

    elif class_name == 'StimuliRegion':
        root_metadata['connectivity'] = GID_PREFIX + root_metadata['connectivity']

        _migrate_stimuli(root_metadata, storage_manager, ['weight'])
        view_model_class = RegionStimulusCreatorModel

    elif class_name == 'StimuliSurface':
        root_metadata['surface'] = GID_PREFIX + root_metadata['surface']

        _migrate_stimuli(root_metadata, storage_manager, ['focal_points_surface', 'focal_points_triangles'])
        view_model_class = SurfaceStimulusCreatorModel

    elif class_name == 'ConnectivityAnnotations':
        root_metadata['connectivity'] = GID_PREFIX + root_metadata['connectivity']
        root_metadata['written_by'] = "tvb.adapters.datatypes.h5.annotation_h5.ConnectivityAnnotationsH5"
        h5_class = ConnectivityAnnotationsH5

        dependent_attributes['connectivity'] = root_metadata['connectivity']
        _migrate_dataset_metadata(['region_annotations'], storage_manager)
        view_model_class = BRCOImporterModel

    elif class_name == 'ValueWrapper':
        root_metadata['data_type'] = root_metadata['data_type'].replace("\"", '')
        root_metadata['written_by'] = "tvb.adapters.datatypes.h5.mapped_value_h5.ValueWrapperH5"
        h5_class = ValueWrapperH5
        view_model_class = BaseBCTModel

    root_metadata['operation_tag'] = ''
    storage_manager.set_metadata(root_metadata)


    if storage_migrate is False:
        return

    with h5_class(input_file) as f:

        # Create the corresponding datatype to be stored in db
        datatype = REGISTRY.get_datatype_for_h5file(f)()
        f.load_into(datatype)
        generic_attributes = f.load_generic_attributes()
        datatype_index = REGISTRY.get_index_for_datatype(datatype.__class__)()

        # Get the dependent datatypes
        for attr_name, attr_value in dependent_attributes.items():
            dependent_datatype_index = dao.get_datatype_by_gid(attr_value.replace('-', '').replace('urn:uuid:', ''))
            dependent_datatype = h5.load_from_index(dependent_datatype_index)
            setattr(datatype, attr_name, dependent_datatype)

        # Check if it's needed to create a ViewModel H5
        files_in_op_dir = os.listdir(os.path.join(input_file, os.pardir))
        has_vm = False
        if len(files_in_op_dir) > 2:
            for file in files_in_op_dir:
                if view_model_class.__name__ in file:
                    has_vm = True
                    break

        if has_vm is False:

            # Get the VM attributes and the operation parameters from the version 4 DB
            view_model = view_model_class()
            view_model.generic_attributes = generic_attributes

            operation = dao.get_operation_by_id(op_id)
            vm_attributes = [i for i in view_model_class.__dict__.keys() if i[:1] != '_']
            op_parameters = eval(operation.view_model_gid)

            # Create a new burst if datatype is a TimeSeries and assign it
            if 'parent_burst_id' in op_parameters:
                burst_gid = _create_new_burst(operation.fk_launched_in, root_metadata)
                generic_attributes.parent_burst = burst_gid
                storage_manager.set_metadata(root_metadata)

            # Get parent_burst
            if 'time_series' in op_parameters:
                ts = dao.get_datatype_by_gid(op_parameters['time_series'].replace('-', '').replace('urn:uuid:', ''))
                root_metadata['parent_burst'] = GID_PREFIX + ts.fk_parent_burst
                storage_manager.set_metadata(root_metadata)

            # Set the view model attributes
            for attr in vm_attributes:
                if attr not in changed_values:
                    if attr in op_parameters:
                        try:
                            setattr(view_model, attr, op_parameters[attr])
                        except (TraitTypeError, TraitAttributeError):
                            pass
                else:
                    setattr(view_model, attr, changed_values[attr])

            # Store the ViewModel and the Operation
            h5.store_view_model(view_model, os.path.dirname(input_file))
            operation.view_model_gid = view_model.gid.hex
            dao.store_entity(operation)

        # Populate datatype
        datatype_index.fill_from_has_traits(datatype)
        datatype_index.fill_from_generic_attributes(generic_attributes)
        datatype_index.fk_from_operation = op_id

    # Finally store new datatype in db
    dao.store_entity(datatype_index)
