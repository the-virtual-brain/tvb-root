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
Utility functions for using siibra to extract Structural and Functional connectivities

.. moduleauthor:: Romina Baila <romina.baila@codemart.ro>
"""
import numpy as np
import siibra
from enum import Enum
from tvb.basic.logger.builder import get_logger
from tvb.datatypes import connectivity
from tvb.datatypes.graph import ConnectivityMeasure

LOGGER = get_logger(__name__)

DEFAULT_ATLAS = 'Multilevel Human Atlas'
DEFAULT_PARCELLATION = 'Julich-Brain Cytoarchitectonic Maps 2.9'
DEFAULT_COHORT = 'HCP'
THOUSAND_BRAINS_COHORT = '1000BRAINS'


class Component2Modality(Enum):
    WEIGHTS = siibra.features.connectivity.StreamlineCounts
    TRACTS = siibra.features.connectivity.StreamlineLengths
    FUNCTIONAL_CONNECTIVITY = siibra.features.connectivity.FunctionalConnectivity


# ########################################## SIIBRA CREATOR INITIALIZATION #############################################
def get_cohorts_for_sc(parcellation_name):
    """
    Given a parcellation name, return all the cohorts related to it and containing data about Structural Connectivities.
    We chose to return the options for Struct. Conn., as the Functional Conn. are anyway retrieved
    only together with Structural Conn.
    """
    parcellation = siibra.parcellations[parcellation_name]
    features = siibra.features.get(parcellation, siibra.features.connectivity.StreamlineCounts)
    return [f.cohort.upper() for f in features]


# ######################################## SIIBRA PARAMETERS CONFIGURATION #############################################
def check_atlas_parcellation_compatible(atlas, parcellation):
    """ Given an atlas and a parcellation, verify that the atlas contains the parcellation, i.e. they are compatible """
    return parcellation in list(atlas.parcellations)


def get_atlases_for_parcellation(parcelation):
    """ Given a parcelation, return all the atlases that contain it """
    atlases = siibra.atlases
    atlases = [a for a in atlases if parcelation in list(a.parcellations)]
    return atlases


def get_parcellations_for_atlas(atlas):
    """ Given the name of an atlas, return all the parcellations inside it """
    return list(atlas.parcellations)


def parse_subject_ids(subject_ids, cohort):
    """
    Given a string representing subject ids or a range of subject ids; return the list containing all the included ids
    """
    parsed_ids_as_str = []
    individual_splits = subject_ids.split(';')

    if cohort == THOUSAND_BRAINS_COHORT:
        for s in individual_splits:
            parsed_ids_as_str.append(s)
    elif cohort == DEFAULT_COHORT:
        parsed_ids = []
        for s in individual_splits:
            # if a range was written
            if '-' in s:
                start, end = s.split('-')
                start_int = int(start)
                end_int = int(end) + 1  # so that the last element in range is also included
                ids_list_from_range = list(range(start_int, end_int))
                parsed_ids.extend(ids_list_from_range)
            else:
                s_int = int(s)
                parsed_ids.append(s_int)

        # convert the subject ids list back to strings into the required format
        parsed_ids_as_str = [str(id).zfill(3) for id in parsed_ids]

    return parsed_ids_as_str


def init_siibra_params(atlas_name, parcellation_name, cohort_name, subject_ids):
    # check that the atlas and the parcellation exist within siibra
    atlas = siibra.atlases[atlas_name] if atlas_name else None
    parcellation = siibra.parcellations[parcellation_name] if parcellation_name else None

    # check compatibility of atlas and parcellation
    if atlas and parcellation:
        compatible = check_atlas_parcellation_compatible(atlas, parcellation)
        if not compatible:
            raise ValueError(f'Atlas \'{atlas.name}\' does not contain parcellation \'{parcellation.name}\'. '
                             f'Please choose a different atlas and/or parcellation.')

    if atlas and not parcellation:
        LOGGER.warning(f'No parcellation was provided, so a default one will be selected.')
        parcellations = get_parcellations_for_atlas(atlas)
        no_parcellations = len(parcellations)
        if no_parcellations < 1:
            raise AttributeError(f'No default parcellation was found for atlas {atlas.name}!')
        if no_parcellations > 1:
            LOGGER.info(
                f'Multiple parcellations were found for atlas {atlas.name}. An arbitrary one will be selected.')
        parcellation = parcellations[0]

    if not atlas and parcellation:
        LOGGER.warning('A parcellation was provided without an atlas, so a default atlas will be selected.')
        atlases = get_atlases_for_parcellation(parcellation)
        no_atlases = len(atlases)
        if no_atlases < 1:
            raise AttributeError(f'No default atlas containing parcellation {parcellation.name} was found!')
        if no_atlases > 1:
            LOGGER.info(
                f'Multiple atlases containing parcellation {parcellation_name} were found. '
                f'An arbitrary one will be selected')
        atlas = atlases[0]

    if not atlas and not parcellation:
        LOGGER.warning(f'No atlas and no parcellation were provided, so default ones will be selected.')
        atlas = siibra.atlases[DEFAULT_ATLAS]
        parcellation = siibra.parcellations[DEFAULT_PARCELLATION]

    LOGGER.info(f'Using atlas {atlas.name} and parcellation {parcellation.name}')

    # check the compatibility of cohort and parcellation
    cohort_options = get_cohorts_for_sc(parcellation.name)
    if cohort_name is None:
        cohort_name = DEFAULT_COHORT
    elif cohort_name not in cohort_options:
        raise ValueError(f'The cohort \"{cohort_name}\" is not available for parcellation \"{parcellation.name}\"!')

    # check subject ids
    if not subject_ids:
        LOGGER.info(
            f'The mean across all connectivities from cohort {cohort_name} will be computed.')
        subject_ids = [None]
    else:
        subject_ids = parse_subject_ids(subject_ids, cohort_name)

    return atlas, parcellation, cohort_name, subject_ids


# ############################################# CONNECTIVITY METHODS ###################################################
def get_hemispheres_for_regions(region_names):
    """ Given a list of region names, compute the hemispheres to which they belon to """
    LOGGER.info(f'Computing hemispheres for regions')
    hemi = []
    for name in region_names:
        if 'right' in name:
            hemi.append(1)
        # TODO: regions referring to both hemispheres are put in the left hemisphere; change this?
        else:
            hemi.append(0)

    return hemi


def get_regions_positions(regions):
    """
    Given a list of siibra regions, compute the positions of their centroids.
    :param: regions - list of siibra region objects
    :return: positions - list of tuples; each tuple represents the position of a region in `regions` and contains
    3 floating point coordinates
    """
    LOGGER.info(f'Computing positions for regions')
    positions = []
    space = siibra.spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC  # commonly used space in the documentation

    for r in regions:
        centroid = r.spatial_props(space)['components'][0]['centroid'].coordinate
        positions.append(centroid)

    return positions


# ###################################### STRUCTURAL CONNECTIVITY METHODS ###############################################
def get_connectivity_matrix(parcellation, cohort, subjects, component):
    # type: (siibra.core.parcellation.Parcellation, str, list, Component2Modality) -> {}
    """
    Retrieve the structural connectivity components (weights/tracts) for all the subjects provided, for the specified
    parcellation and cohort. The matrices are returned inside a dictionary, where the keys are the subject ids and the
    values represent the connectivity matrix.
    :param: parcellation - siibra Parcellation object for which we compute the connectivity matrices
    :param: cohort - name of cohort for which we compute the connectivity matrices
    :param: subjects - list containing the subject ids as strings
    :param: component - enum value specifying the connectivity component we want, weights or tracts
    return: conn_matrices - dict containing the conn. matrices (values) for the specified subject ids (keys)
    """
    modality = component.value
    features = siibra.features.get(parcellation, modality)
    conn_for_cohort = None  # conn. obj. (StreamlineCounts/StreamlineLengths) for specified cohort
    conn_matrices = {}  # dict containing connectivity matrices for the specified users
    # select the connectivities for the specific cohort
    for f in features:
        if f.cohort == cohort:
            conn_for_cohort = f

    # for 1000BRAINS cohort, if the user did not specify a suffix (_1, _2), get all the possible ids for that subject
    if cohort == THOUSAND_BRAINS_COHORT and subjects is not None:
        subjects = [s for s in sorted(conn_for_cohort.subjects) if any(x in s for x in subjects)]

    # get the conn. matrices for each specified subject
    for s in subjects:
        if s is None:
            s = 'mean'
        matrix = conn_for_cohort.get_matrix(s)
        conn_matrices[s] = matrix
        LOGGER.info(f'{component.name} for subject {s} retrieved successfully.')

    return conn_matrices


def create_tvb_structural_connectivity(weights_matrix, tracts_matrix, region_names, hemispheres, positions):
    """ Compute a TVB Connectivity based on its components obtained from siibra """

    conn = connectivity.Connectivity()
    conn.weights = weights_matrix.to_numpy()
    conn.tract_lengths = tracts_matrix.to_numpy()
    conn.region_labels = np.array(region_names)
    conn.hemispheres = np.array(hemispheres, dtype=np.bool_)
    conn.centres = np.array(positions)

    conn.configure()
    return conn


def get_structural_connectivities_from_kg(atlas=None, parcellation=None, cohort=None, subject_ids=None):
    """
    Return a list of TVB Structural Connectivities, based on the specified atlas and parcellation,
    for the subjects mentioned in 'subject_ids'
    """
    atlas, parcellation, cohort, subject_ids = init_siibra_params(atlas, parcellation, cohort, subject_ids)
    connectivities = {}
    weights = get_connectivity_matrix(parcellation, cohort, subject_ids, Component2Modality.WEIGHTS)
    tracts = get_connectivity_matrix(parcellation, cohort, subject_ids, Component2Modality.TRACTS)

    # regions are the same for all weights and tract lengths matrices, so they can be computed only once
    regions = list(weights.values())[0].index.values
    # because siibra sometimes returns tuples instead of actual regions, change list to contain only regions
    regions = [r[1] if type(r) == tuple else r for r in regions]
    region_names = [r.name for r in regions]
    hemi = get_hemispheres_for_regions(region_names)
    positions = get_regions_positions(regions)

    LOGGER.info(f'Computing TVB Connectivities')
    for subject, matrix in weights.items():
        weights_matrix = matrix
        tracts_matrix = tracts[subject]
        tvb_conn = create_tvb_structural_connectivity(weights_matrix, tracts_matrix, region_names, hemi, positions)

        # structural connectivities stored as dict, to link a functional connectivity with the correct
        # structural connectivity when creating connectivity measures
        connectivities[subject] = tvb_conn

    return connectivities


# #################################### FUNCTIONAL CONNECTIVITY METHODS #################################################
def get_functional_connectivity_matrix(parcellation, cohort, subject):
    """
    Given a list of functional connectivities and a list of subject ids, keep only the functional connectivities
    for those subjects
    """
    modality = Component2Modality.FUNCTIONAL_CONNECTIVITY.value
    features = siibra.features.get(parcellation, modality)
    fcs_list = []  # the FC matrices
    fcs_names_list = []  # the name of the files containing the FC matrices in fcs_list

    for f in features:
        if f.cohort == cohort:
            fc_matrix = f.get_matrix(subject)
            fcs_names_list.append(f._files[subject])
            fcs_list.append(fc_matrix)

    return fcs_list, fcs_names_list


def get_fc_name_from_file_path(path_to_file):
    """
    Given the entire path to a file containing a siibra FunctionalConnectivity, return just the filename
    Note: highly dependent on EBRAINS/siibra storage conventions
    """
    file_with_extension = path_to_file.rsplit('/', 1)[1]
    filename = file_with_extension.rsplit('.', 1)[0]

    return filename


def create_tvb_connectivity_measure(siibra_fc, structural_connectivity, siibra_fc_filename):
    """
    Given a FunctionalConnectivity from  siibra for a subject and its corresponding TVB Structural Connectivity
    (for the same subject), return a TVB ConnectivityMeasure having as data the siibra FC
    """
    fc_matrix = siibra_fc.to_numpy()
    conn_measure = ConnectivityMeasure(array_data=fc_matrix, connectivity=structural_connectivity)
    conn_measure.title = get_fc_name_from_file_path(siibra_fc_filename)

    return conn_measure


def get_connectivity_measures_from_kg(atlas=None, parcellation=None, cohort=None, subject_ids=None,
                                      structural_connectivities=None):
    atlas, parcellation, cohort, subject_ids = init_siibra_params(atlas, parcellation, cohort, subject_ids)
    conn_measures = {}

    # for 1000BRAINS cohort, if the user did not specify a suffix (_1, _2), get all the possible ids for that subject
    if cohort == THOUSAND_BRAINS_COHORT and any('_' not in s for s in subject_ids):
        f = [f for f in siibra.features.get(parcellation, siibra.features.connectivity.FunctionalConnectivity)
             if f.cohort == THOUSAND_BRAINS_COHORT]
        f = f[0]  # get the first feature from the feature list, we just want to know the subject names
        subjects_for_cohort = f.subjects
        subject_ids = [s for s in sorted(subjects_for_cohort) if any(x in s for x in subject_ids)]

    for s in subject_ids:
        if s is None:
            s = 'mean'
        conn_measures[s] = []
        sc = structural_connectivities[s]

        fcs, fcs_names = get_functional_connectivity_matrix(parcellation, cohort, s)

        # create a tvb conn measure from a siibra fc and append it in the final dict for the specified subject
        for i in range(len(fcs)):
            conn_measure = create_tvb_connectivity_measure(fcs[i], sc, fcs_names[i])
            conn_measures[s].append(conn_measure)

    return conn_measures


# ################################################# FINAL API ##########################################################
def get_connectivities_from_kg(atlas=None, parcellation=None, cohort=DEFAULT_COHORT,
                               subject_ids=None, compute_fc=False):
    """
    Compute the TVB Structural Connectivities and optionally Functional Connectivities for the selected subjects
    """
    conn_measures_dict = {}
    sc_dict = get_structural_connectivities_from_kg(atlas, parcellation, cohort, subject_ids)

    if compute_fc:
        conn_measures_dict = get_connectivity_measures_from_kg(atlas, parcellation, cohort, subject_ids, sc_dict)

    return sc_dict, conn_measures_dict
