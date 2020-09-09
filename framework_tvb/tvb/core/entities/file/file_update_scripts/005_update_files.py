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
import os
import sys
import numpy
from tvb.core.neocom.h5 import REGISTRY
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.exceptions import IncompatibleFileManagerException
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neotraits._h5accessors import DataSetMetaData
from tvb.core.neotraits._h5core import H5File
from tvb.core.neotraits.h5 import STORE_STRING

LOGGER = get_logger(__name__)
FIELD_SURFACE_MAPPING = "has_surface_mapping"
FIELD_VOLUME_MAPPING = "has_volume_mapping"


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


def _bytes_ds_to_string_ds(storage_manager, ds_name):
    bytes_labels = storage_manager.get_data(ds_name)
    string_labels = []
    for i in range(len(bytes_labels)):
        string_labels.append(str(bytes_labels[i], 'utf-8'))

    storage_manager.remove_data(ds_name)
    storage_manager.store_data(ds_name, numpy.asarray(string_labels).astype(STORE_STRING))
    return storage_manager

# def _set_parent_burst():



def _migrate_dataset_metadata(dataset_list, storage_manager):
    for dataset in dataset_list:
        conn_metadata = DataSetMetaData.from_array(storage_manager.get_data(dataset)).to_dict()
        storage_manager.set_metadata(conn_metadata, dataset)
        metadata = storage_manager.get_metadata(dataset)
        if 'Variance' in metadata:
            storage_manager.remove_metadata('Variance', dataset)
        if 'Size' in metadata:
            storage_manager.remove_metadata('Size',dataset)


def update(input_file):
    """
    :param input_file: the file that needs to be converted to a newer file storage version.
    """

    if not os.path.isfile(input_file):
        raise IncompatibleFileManagerException("Not yet implemented update for file %s" % input_file)

    folder, file_name = os.path.split(input_file)
    storage_manager = HDF5StorageManager(folder, file_name)

    root_metadata = storage_manager.get_metadata()

    if DataTypeMetaData.KEY_CLASS_NAME not in root_metadata:
        raise IncompatibleFileManagerException("File %s received for upgrading 4 -> 5 is not valid, due to missing "
                                               "metadata: %s" % (input_file, DataTypeMetaData.KEY_CLASS_NAME))

    lowercase_keys = []
    for key, value in root_metadata.items():
        root_metadata[key] = str(value, 'utf-8')
        lowercase_keys.append(_lowercase_first_character(key))
        storage_manager.remove_metadata(key)

    root_metadata = dict(zip(lowercase_keys, list(root_metadata.values())))
    root_metadata[TvbProfile.current.version.DATA_VERSION_ATTRIBUTE] = TvbProfile.current.version.DATA_VERSION
    class_name = root_metadata["type"]

    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace('datetime:', '')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(':', '-')
    root_metadata[DataTypeMetaData.KEY_DATE] = root_metadata[DataTypeMetaData.KEY_DATE].replace(' ', ',')

    try:
        datatype_class = getattr(sys.modules[root_metadata["module"]],
                             root_metadata["type"])
        h5_class = REGISTRY.get_h5file_for_datatype(datatype_class)
        root_metadata[H5File.KEY_WRITTEN_BY] = h5_class.__module__ + '.' + h5_class.__name__
    except KeyError:
        pass

    root_metadata['user_tag_1'] = ''
    root_metadata['gid'] = "urn:uuid:" + root_metadata['gid']

    root_metadata.pop("type")
    root_metadata.pop("module")
    root_metadata.pop('data_version')

    if 'TimeSeries' in class_name:
        root_metadata.pop(FIELD_SURFACE_MAPPING)
        root_metadata.pop(FIELD_VOLUME_MAPPING)

        root_metadata['nr_dimensions'] = int(root_metadata['nr_dimensions'])
        root_metadata['sample_period'] = float(root_metadata['sample_period'])
        root_metadata['start_time'] = float(root_metadata['start_time'])

        root_metadata["sample_period_unit"] = root_metadata["sample_period_unit"].replace("\"", '')
        root_metadata[DataTypeMetaData.KEY_TITLE] = root_metadata[DataTypeMetaData.KEY_TITLE].replace("\"", '')
        root_metadata['region_mapping'] = "urn:uuid:" + root_metadata['region_mapping']
        root_metadata['connectivity'] = "urn:uuid:" + root_metadata['connectivity']
        root_metadata = _pop_lengths(root_metadata)

    elif 'Connectivity' in class_name and ('Local' not in class_name):
        root_metadata['number_of_connections'] = int(root_metadata['number_of_connections'])
        root_metadata['number_of_regions'] = int(root_metadata['number_of_regions'])

        if root_metadata['undirected'] == "0":
            root_metadata['undirected'] = "bool:False"
        else:
            root_metadata['undirected'] = "bool:True"

        if root_metadata['saved_selection'] == 'null':
            root_metadata['saved_selection'] = '[]'

        _migrate_dataset_metadata(['areas', 'centres', 'cortical', 'hemispheres', 'orientations', 'region_labels',
                                   'tract_lengths', 'weights'], storage_manager)

        storage_manager.remove_metadata('Mean non zero', 'tract_lengths')
        storage_manager.remove_metadata('Min. non zero', 'tract_lengths')
        storage_manager.remove_metadata('Var. non zero', 'tract_lengths')
        storage_manager.remove_metadata('Mean non zero', 'weights')
        storage_manager.remove_metadata('Min. non zero', 'weights')
        storage_manager.remove_metadata('Var. non zero', 'weights')

    elif class_name in ['BrainSkull', 'CorticalSurface', 'SkinAir', 'BrainSkull', 'SkullSkin', 'EEGCap', 'FaceSurface']:
        root_metadata['edge_max_length'] = float(root_metadata['edge_max_length'])
        root_metadata['edge_mean_length'] = float(root_metadata['edge_mean_length'])
        root_metadata['edge_min_length'] = float(root_metadata['edge_min_length'])

        root_metadata['number_of_split_slices'] = int(root_metadata['number_of_split_slices'])
        root_metadata['number_of_triangles'] = int(root_metadata['number_of_triangles'])
        root_metadata['number_of_vertices'] = int(root_metadata['number_of_vertices'])

        root_metadata["surface_type"] = root_metadata["surface_type"].replace("\"", '')

        storage_manager.store_data('split_triangles', [])

        _migrate_dataset_metadata(['split_triangles', 'triangle_normals', 'triangles', 'vertex_normals', 'vertices'],
                                  storage_manager)

        root_metadata['zero_based_triangles'] = "bool:" + root_metadata['zero_based_triangles'][:1].upper() \
                                                + root_metadata['zero_based_triangles'][1:]
        root_metadata['bi_hemispheric'] = "bool:" + root_metadata['bi_hemispheric'][:1].upper() \
                                          + root_metadata['bi_hemispheric'][1:]
        root_metadata['valid_for_simulations'] = "bool:" + root_metadata['valid_for_simulations'][:1].upper() \
                                                 + root_metadata['valid_for_simulations'][1:]

    elif 'RegionMapping' in class_name:
        root_metadata = _pop_lengths(root_metadata)
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')

        root_metadata['surface'] = "urn:uuid:" + root_metadata['surface']
        root_metadata['connectivity'] = "urn:uuid:" + root_metadata['connectivity']

        _migrate_dataset_metadata(['array_data'], storage_manager)
    elif 'Sensors' in class_name:
        root_metadata['number_of_sensors'] = int(root_metadata['number_of_sensors'])
        root_metadata['sensors_type'] = root_metadata["sensors_type"].replace("\"", '')
        root_metadata['has_orientation'] = "bool:" + root_metadata['has_orientation'][:1].upper() \
                                           + root_metadata['has_orientation'][1:]

        storage_manager.remove_metadata('Size', 'labels')
        storage_manager.remove_metadata('Size', 'locations')
        storage_manager.remove_metadata('Variance', 'locations')

        labels_metadata = {'Maximum': '', 'Mean': '', 'Minimum': ''}
        storage_manager.set_metadata(labels_metadata, 'labels')

        storage_manager = _bytes_ds_to_string_ds(storage_manager, 'labels')

        datasets = ['labels', 'locations']

        if 'MEG' in class_name:
            storage_manager.remove_metadata('Size', 'orientations')
            storage_manager.remove_metadata('Variance', 'orientations')
            datasets.append('orientations')

        _migrate_dataset_metadata(datasets, storage_manager)
    elif 'Projection' in class_name:
        root_metadata['written_by'] = "tvb.adapters.datatypes.h5.projections_h5.ProjectionMatrixH5"
        root_metadata['projection_type'] = root_metadata["projection_type"].replace("\"", '')
        root_metadata['sensors'] = "urn:uuid:" + root_metadata['sensors']
        root_metadata['sources'] = "urn:uuid:" + root_metadata['sources']

        storage_manager.remove_metadata('Size', 'projection_data')
        storage_manager.remove_metadata('Variance', 'projection_data')

        _migrate_dataset_metadata(['projection_data'], storage_manager)
    elif 'LocalConnectivity' in class_name:
        root_metadata['cutoff'] = float(root_metadata['cutoff'])
        root_metadata['surface'] = "urn:uuid:" + root_metadata['surface']

        storage_manager.remove_metadata('shape', 'matrix')

        matrix_metadata = storage_manager.get_metadata('matrix')
        matrix_metadata['Shape'] = str(matrix_metadata['Shape'], 'utf-8')
        matrix_metadata['dtype'] = str(matrix_metadata['dtype'], 'utf-8')
        matrix_metadata['format'] = str(matrix_metadata['format'], 'utf-8')
        storage_manager.set_metadata(matrix_metadata, 'matrix')
    elif 'Volume' in class_name:
        root_metadata['voxel_unit'] = root_metadata['voxel_unit'].replace("\"", '')

    if class_name == 'CoherenceSpectrum':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['nfft'] = int(root_metadata['nfft'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        array_data = storage_manager.get_data('array_data')
        storage_manager.remove_data('array_data')
        storage_manager.store_data('array_data', numpy.asarray(array_data, dtype=numpy.float64))
        _migrate_dataset_metadata(['array_data', 'frequency'], storage_manager)

    if  class_name == 'ComplexCoherenceSpectrum':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['epoch_length'] = float(root_metadata['epoch_length'])
        root_metadata['segment_length'] = float(root_metadata['segment_length'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        root_metadata['windowing_function'] = root_metadata['windowing_function'].replace("\"", '')
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        _migrate_dataset_metadata(['array_data', 'cross_spectrum'], storage_manager)

    if class_name == 'WaveletCoefficients':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['q_ratio'] = float(root_metadata['q_ratio'])
        root_metadata['sample_period'] = float(root_metadata['sample_period'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        root_metadata['mother'] = root_metadata['mother'].replace("\"", '')
        root_metadata['normalisation'] = root_metadata['normalisation'].replace("\"", '')

        _migrate_dataset_metadata(['amplitude', 'array_data', 'frequencies', 'phase', 'power'], storage_manager)

    if class_name == 'CrossCorrelation':
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        _migrate_dataset_metadata(['array_data', 'time'], storage_manager)

    if class_name == 'Fcd':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['sp'] = float(root_metadata['sp'])
        root_metadata['sw'] = float(root_metadata['sw'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

    if class_name == 'ConnectivityMeasure':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        _pop_lengths(root_metadata)

        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        root_metadata['connectivity'] = "urn:uuid:" + root_metadata['connectivity']
        _migrate_dataset_metadata(['array_data'], storage_manager)

    if class_name == 'FourierSpectrum':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['segment_length'] = float(root_metadata['segment_length'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        _migrate_dataset_metadata(['amplitude', 'array_data', 'average_power',
                                   'normalised_average_power', 'phase', 'power'], storage_manager)

    if class_name == 'IndependentComponents':
        root_metadata['n_components'] = int(root_metadata['n_components'])
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']

        _migrate_dataset_metadata(['component_time_series', 'mixing_matrix', 'norm_source',
                                   'normalised_component_time_series', 'prewhitening_matrix',
                                   'unmixing_matrix'], storage_manager)

    if class_name == 'CorrelationCoefficients':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        _migrate_dataset_metadata(['array_data'], storage_manager)

    if class_name == 'PrincipalComponents':
        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        _migrate_dataset_metadata(['component_time_series', 'fractions',
                                   'norm_source', 'normalised_component_time_series',
                                   'weights'], storage_manager)

    if class_name == 'Covariance':
        root_metadata.pop('aggregation_functions')
        root_metadata.pop('dimensions_labels')
        root_metadata.pop('nr_dimensions')
        root_metadata.pop('label_x')
        root_metadata.pop('label_y')
        root_metadata.pop(DataTypeMetaData.KEY_TITLE)
        _pop_lengths(root_metadata)

        root_metadata['source'] = "urn:uuid:" + root_metadata['source']
        _migrate_dataset_metadata(['array_data'], storage_manager)

    if class_name == 'DatatypeMeasure':
        root_metadata['written_by'] = 'tvb.core.entities.file.simulator.datatype_measure_h5.DatatypeMeasureH5'

    if class_name == 'StimuliRegion':
        root_metadata['connectivity'] = "urn:uuid:" + root_metadata['connectivity']

    if class_name == 'StimuliSurface':
        root_metadata['surface'] = "urn:uuid:" + root_metadata['surface']

    root_metadata['operation_tag'] = ''
    storage_manager.set_metadata(root_metadata)