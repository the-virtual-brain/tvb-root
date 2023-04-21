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
from nibabel import Nifti1Image
from tvb.basic.logger.builder import get_logger
from tvb.datatypes import connectivity
from tvb.datatypes.graph import ConnectivityMeasure

LOGGER = get_logger(__name__)

DEFAULT_ATLAS = 'Multilevel Human Atlas'
DEFAULT_PARCELLATION = 'Julich-Brain Cytoarchitectonic Maps 2.9'


class Component2Modality(Enum):
    WEIGHTS = siibra.modalities.StreamlineCounts
    TRACTS = siibra.modalities.StreamlineLengths
    FC = siibra.modalities.FunctionalConnectivity


# ######################################## SIIBRA PARAMETERS CONFIGURATION #############################################
def check_atlas_parcellation_compatible(atlas, parcellation):
    """ Given an atlas and a parcellation, verify that the atlas contains the parcellation, i.e. they are compatible """
    return parcellation in list(atlas.parcellations)


def get_atlases_for_parcellation(parcelation):
    """ Given a parcelation, return all the atlases that contain it """
    return list(parcelation.atlases)


def get_parcellations_for_atlas(atlas):
    """ Given the name of an atlas, return all the parcellations inside it """
    return list(atlas.parcellations)


def parse_subject_ids(subject_ids):
    """
    Given a string representing subject ids or a range of subject ids; return the list containing all the included ids
    """
    parsed_ids = []
    individual_splits = subject_ids.split(';')
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
    parsed_ids_strings = [str(id).zfill(3) for id in parsed_ids]
    return parsed_ids_strings


def init_siibra_params(atlas_name, parcellation_name, subject_ids):
    atlas = siibra.atlases[atlas_name] if atlas_name else None
    parcellation = siibra.parcellations[parcellation_name] if parcellation_name else None

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

    if not subject_ids:
        LOGGER.info(
            f'No list of subject ids was provided, so the connectivities will be computed for all available subjects!')
        subject_ids = 'all'
    elif subject_ids != 'all':
        subject_ids = parse_subject_ids(subject_ids)

    return atlas, parcellation, subject_ids


# ################################################# SIIBRA METHODS #####################################################
def compute_centroids(region, space):
    """
    Compute the centroids of a region in a specified space.
    It is the uncached and slightly modified version of siibra.core.region.spatial_props.
    """
    from skimage import measure

    if not region.mapped_in_space(space):
        raise RuntimeError(
            f"Spatial properties of {region.name} cannot be computed in {space.name}. "
            "This region is only mapped in these spaces: "
            ", ".join(s.name for s in region.supported_spaces)
        )

    # build binary mask of the image
    pimg = build_mask_for_centroids(region=region, space=space)

    # compute properties of labelled volume
    A = np.asarray(pimg.get_fdata(), dtype=np.int32).squeeze()
    C = measure.label(A)

    # compute centroid only for the first connected component of the region
    nonzero = np.c_[np.nonzero(C == 1)]
    centroid = siibra.Point(np.dot(pimg.affine, np.r_[nonzero.mean(0), 1])[:3], space=space)
    # tuple() gives the coordinate of the centroid
    centroid_coord = tuple(centroid)

    return centroid_coord


def build_mask_for_centroids(region, space, resolution_mm=None, maptype: siibra.MapType = siibra.MapType.LABELLED,
                             threshold_continuous=None, consider_other_types=True):
    """
    Returns a mask for the specified region in the specified space.
    It is the uncached version of siibra.core.region.build_mask
    """
    spaceobj = siibra.core.Space.REGISTRY[space]
    if spaceobj.is_surface:
        raise NotImplementedError(
            "Region masks for surface spaces are not yet supported."
        )

    mask = None
    if isinstance(maptype, str):
        maptype = siibra.MapType[maptype.upper()]

    if region.has_regional_map(spaceobj, maptype):
        # the region has itself a map of that type linked
        mask = region.get_regional_map(space, maptype).fetch(
            resolution_mm=resolution_mm
        )
    else:
        # retrieve  map of that type from the region's corresponding parcellation map
        parcmap = region.parcellation.get_map(spaceobj, maptype)
        mask = parcmap.fetch_regionmap(region, resolution_mm=resolution_mm)

    if mask is None:
        # Attempt to produce a map from the child regions.
        # We only accept this if all child regions produce valid masks.
        # NOTE We ignore extension regions here, since this is a complex case currently (e.g. iam regions in BigBrain)
        LOGGER.debug(f"Merging child regions to build mask for their parent {region.name}:")
        maskdata = None
        affine = None
        for c in region.children:
            if c.extended_from is not None:
                continue
            childmask = build_mask_for_centroids(c, space, resolution_mm, maptype, threshold_continuous)
            if childmask is None:
                LOGGER.warning(f"No success getting mask for child {c.name}")
                break
            if maskdata is None:
                affine = childmask.affine
                maskdata = np.asanyarray(childmask.dataobj)
            else:
                assert (childmask.affine == affine).all()
                maskdata = np.maximum(maskdata, np.asanyarray(childmask.dataobj))
        else:
            # we get here only if the for loop was not interrupted by 'break'
            if maskdata is not None:
                return Nifti1Image(maskdata, affine)

    if mask is None:
        # No map of the requested type found for the region.
        LOGGER.warn(
            f"Could not compute {maptype.name.lower()} mask for {region.name} in {spaceobj.name}."
        )
        if consider_other_types:
            for other_maptype in (set(siibra.MapType) - {maptype}):
                mask = region.build_mask(
                    space, resolution_mm, other_maptype, threshold_continuous, consider_other_types=False
                )
                if mask is not None:
                    LOGGER.info(
                        f"A mask was generated from map type {other_maptype.name.lower()} instead."
                    )
                    return mask
        return None

    if (threshold_continuous is not None) and (maptype == siibra.MapType.CONTINUOUS):
        data = np.asanyarray(mask.dataobj) > threshold_continuous
        LOGGER.info(f"Mask built using a continuous map thresholded at {threshold_continuous}.")
        assert np.any(data > 0)
        mask = Nifti1Image(data.astype("uint8").squeeze(), mask.affine)

    return mask


# ######################################## COMMON CONNECTIVITY METHODS #################################################
def get_connectivity_component(parcellation, component):
    # type: (str, Component2Modality) -> []
    """
        :return: a list of all available connectivity components (weights/tract lengths)
    """
    modality = component.value
    all_conns = siibra.get_features(parcellation, modality)

    if len(all_conns) == 0:
        raise AttributeError(f'No connectivity component \'{Component2Modality(component).name}\' was found '
                             f'in parcellation \'{parcellation}\'!')
    return all_conns


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
    """ Given a list of regions, compute the positions of their centroids """
    LOGGER.info(f'Computing positions for regions')
    positions = []
    space = siibra.spaces.MNI152_2009C_NONL_ASYM  # commonly used space in the documentation

    for r in regions:
        centroid = compute_centroids(r, space)
        positions.append(centroid)

    return positions


# ###################################### STRUCTURAL CONNECTIVITY METHODS ###############################################
def filter_structural_connectivity_by_id(weights, tracts, subj_ids):
    """
    Given two lists of connectivity weights and tract lengths and a list of subject ids, keep only the weights and
    tracts for those subjects
    """
    filtered_weights = []
    filtered_tracts = []

    for subj in subj_ids:
        weight = [w for w in weights if w.subject == subj]
        tract = [t for t in tracts if t.subject == subj]

        filtered_weights += weight
        filtered_tracts += tract

    return filtered_weights, filtered_tracts


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


def get_tvb_connectivities_from_kg(atlas=None, parcellation=None, subject_ids=None):
    """
    Return a list of TVB Structural Connectivities, based on the specified atlas and parcellation,
    for the subjects mentioned in 'subject_ids'
    """
    atlas, parcellation, subject_ids = init_siibra_params(atlas, parcellation, subject_ids)
    connectivities = {}
    weights = get_connectivity_component(parcellation, Component2Modality.WEIGHTS)
    tracts = get_connectivity_component(parcellation, Component2Modality.TRACTS)

    if not weights or not tracts:
        raise AttributeError(f'Could not find both weights and tract lengths from parcellation \'{parcellation.name}\', '
                             f'so a connectivity cannot be computed')

    if subject_ids != 'all':
        weights, tracts = filter_structural_connectivity_by_id(weights, tracts, subject_ids)

    # regions are the same for all weights and tract lengths matrices, so they can be computed only once
    regions = weights[0].matrix.index.values
    region_names = [r.name for r in regions]
    hemi = get_hemispheres_for_regions(region_names)
    positions = get_regions_positions(regions)

    LOGGER.info(f'Computing TVB Connectivities')
    no_conns = min(len(weights), len(tracts))  # in siibra v0.3a24 weights have additional subjects
    for i in range(no_conns):
        weights_matrix = weights[i].matrix
        tracts_matrix = tracts[i].matrix
        conn = create_tvb_structural_connectivity(weights_matrix, tracts_matrix, region_names, hemi, positions)
        subj = weights[i].subject

        # structural connectivities stored as dict, to link a functional connectivity with the correct
        # structural connectivity when creating connectivity measures
        connectivities[subj] = conn

    return connectivities


# #################################### FUNCTIONAL CONNECTIVITY METHODS #################################################

def filter_functional_connectivity_by_id(fcs, subj_ids):
    """
    Given a list of functional connectivities and a list of subject ids, keep only the functional connectivities
    for those subjects
    """
    filtered_fcs = []

    for subj in subj_ids:
        fc = [f for f in fcs if f.subject == subj]
        filtered_fcs += fc

    return filtered_fcs


def get_fc_name_from_file_path(path_to_file):
    """
    Given the entire path to a file containing a siibra FunctionalConnectivity, return just the filename
    Note: highly dependent on EBRAINS/siibra storage conventions
    """
    file_with_extension = path_to_file.rsplit('/', 1)[1]
    filename = file_with_extension.rsplit('.', 1)[0]

    return filename


def create_tvb_connectivity_measure(siibra_fc, structural_connectivity):
    """
    Given a FunctionalConnectivity from  siibra for a subject and its corresponding TVB Structural Connectivity
    (for the same subject), return a TVB ConnectivityMeasure having as data the siibra FC
    """
    fc_matrix = siibra_fc.matrix.to_numpy()
    conn_measure = ConnectivityMeasure(array_data=fc_matrix, connectivity=structural_connectivity)
    fc_name = get_fc_name_from_file_path(siibra_fc.filename)
    conn_measure.title = fc_name

    return conn_measure


def get_connectivity_measures_from_kg(atlas=None, parcellation=None, subject_ids=None, structural_connectivities=None):
    atlas, parcellation, subject_ids = init_siibra_params(atlas, parcellation, subject_ids)
    conn_measures = {}

    fcs = get_connectivity_component(parcellation, Component2Modality.FC)

    if not fcs:
        LOGGER.error(
            f'Could not find any functional connectivity in parcellation {parcellation.name}, so a TVB Functional '
            f'Connectivity cannot be computed')
        return

    if subject_ids != 'all':
        fcs = filter_functional_connectivity_by_id(fcs, subject_ids)

    for fc in fcs:
        subject = fc.subject

        # the conn measures are kept in a list, as there are multiple conn measures for the same subject
        if subject not in conn_measures.keys():
            conn_measures[subject] = []
        sc = structural_connectivities[subject]
        conn_measure = create_tvb_connectivity_measure(fc, sc)

        # conn. measures kept as dict to be able to set the subject on GenericAttributes in SiibraCreator
        conn_measures[subject].append(conn_measure)

    return conn_measures


# ################################################# FINAL API ##########################################################


def get_connectivities_from_kg(atlas=None, parcellation=None, subject_ids=None, compute_fc=False):
    """
    Compute the TVB Structural Connectivities and optionally Functional Connectivities for the selected subjects
    """
    conn_measures_dict = {}
    sc_dict = get_tvb_connectivities_from_kg(atlas, parcellation, subject_ids)

    if compute_fc:
        conn_measures_dict = get_connectivity_measures_from_kg(atlas, parcellation, subject_ids, sc_dict)

    return sc_dict, conn_measures_dict
