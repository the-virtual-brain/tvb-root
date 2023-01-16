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
The adapters in this module create new connectivities from the Allen Institute
data using their SDK.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os.path
import numpy
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Float, Int, EnumAttr, TVBEnum
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import SelectField, FloatField
from tvb.core.neotraits.view_model import ViewModel
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.volumes import Volume

LOGGER = get_logger(__name__)


class ResolutionOptionsEnum(TVBEnum):
    TWENTY_FIVE = 25
    FIFTY = 50
    ONE_HUNDRED = 100


class WeightsOptionsEnum(TVBEnum):
    PROJECTION_DENSITY_INJECTION_DENSITY = 1
    PROJECTION_DENSITY = 2
    PROJECTION_ENERGY = 3


class AllenConnectModel(ViewModel):
    resolution = EnumAttr(
        label="Spatial resolution (micron)",
        default=ResolutionOptionsEnum.ONE_HUNDRED,
        doc="""resolution""")

    weighting = EnumAttr(
        label="Definition of the weights of the connectivity :",
        default=WeightsOptionsEnum.PROJECTION_DENSITY_INJECTION_DENSITY,
        doc="""
            1: download injection density <br/>
            2: download projection density <br/>
            3: download projection energy <br/>
            """)

    inj_f_thresh = Float(
        label="Injected percentage of voxels in the inj site",
        default=80,
        required=True,
        doc="""To build the volume and the connectivity select only the areas that have a volume 
        greater than (micron^3): """)

    vol_thresh = Float(
        label="Min volume",
        default=1000000000,
        required=True,
        doc="""To build the connectivity select only the experiment where the percentage of infected voxels 
        in the injection structure is greater than: """)


class AllenConnectomeBuilderForm(ABCAdapterForm):

    def __init__(self):
        super(AllenConnectomeBuilderForm, self).__init__()
        self.resolution = SelectField(AllenConnectModel.resolution)
        self.weighting = SelectField(AllenConnectModel.weighting)
        self.inj_f_thresh = FloatField(AllenConnectModel.inj_f_thresh)
        self.vol_thresh = FloatField(AllenConnectModel.vol_thresh)

    @staticmethod
    def get_view_model():
        return AllenConnectModel

    @staticmethod
    def get_required_datatype():
        return None

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return None


class AllenConnectomeBuilder(ABCAdapter):
    """Handler for uploading a mouse connectivity from Allen dataset using AllenSDK."""

    _ui_name = "Allen connectivity builder"
    _ui_description = "Import mouse connectivity from Allen database (tracer experiments)"

    def get_form_class(self):
        return AllenConnectomeBuilderForm

    def get_output(self):
        return [ConnectivityIndex, VolumeIndex, RegionVolumeMappingIndex, StructuralMRIIndex]

    def launch(self, view_model):
        resolution = view_model.resolution.value
        weighting = view_model.weighting.value
        inj_f_thresh = view_model.inj_f_thresh / 100.
        vol_thresh = view_model.vol_thresh

        project = dao.get_project_by_id(self.current_project_id)
        manifest_file = self.storage_interface.get_allen_mouse_cache_folder(project.name)
        manifest_file = os.path.join(manifest_file, 'mouse_connectivity_manifest.json')
        cache = MouseConnectivityCache(resolution=resolution, manifest_file=manifest_file)

        # the method creates a dictionary with information about which experiments need to be downloaded
        ist2e = dictionary_builder(cache, False)

        # the method downloads experiments necessary to build the connectivity
        projmaps = download_an_construct_matrix(cache, weighting, ist2e, False)

        # the method cleans the file projmaps in 4 steps
        projmaps = pms_cleaner(projmaps)

        # download from the AllenSDK the annotation volume, the template volume
        vol, annot_info = cache.get_annotation_volume()
        template, template_info = cache.get_template_volume()

        # rotate template in the TVB 3D reference:
        template = rotate_reference(template)

        # grab the StructureTree instance
        structure_tree = cache.get_structure_tree()

        # the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
        projmaps = areas_volume_threshold(cache, projmaps, vol_thresh, resolution)

        # the method exclude from the experimental dataset
        # those exps where the injected fraction of pixel in the injection site is lower than than the inj_f_thr 
        projmaps = infected_threshold(cache, projmaps, inj_f_thresh)

        # the method creates file order and keyword that will be the link between the SC order and the
        # id key in the Allen database
        [order, key_ord] = create_file_order(projmaps, structure_tree)

        # the method builds the Structural Connectivity (SC) matrix
        structural_conn = construct_structural_conn(projmaps, order, key_ord)

        # the method returns the coordinate of the centres and the name of the brain areas in the selected parcellation
        [centres, names] = construct_centres(cache, order, key_ord)

        # the method returns the tract lengths between the brain areas in the selected parcellation
        tract_lengths = construct_tract_lengths(centres)

        # the method associated the parent and the grandparents to the child in the selected parcellation with
        # the biggest volume
        [unique_parents, unique_grandparents] = parents_and_grandparents_finder(cache, order, key_ord, structure_tree)

        # the method returns a volume indexed between 0 and N-1, with N=tot brain areas in the parcellation.
        # -1=background and areas that are not in the parcellation
        vol_parcel = mouse_brain_visualizer(vol, order, key_ord, unique_parents, unique_grandparents,
                                            structure_tree, projmaps)

        # results: Connectivity, Volume & RegionVolumeMapping
        # Connectivity
        result_connectivity = Connectivity()
        result_connectivity.centres = centres
        result_connectivity.region_labels = numpy.array(names)
        result_connectivity.weights = structural_conn
        result_connectivity.tract_lengths = tract_lengths
        result_connectivity.configure()
        # Volume
        result_volume = Volume()
        result_volume.origin = numpy.array([[0.0, 0.0, 0.0]])
        result_volume.voxel_size = numpy.array([resolution, resolution, resolution])
        # result_volume.voxel_unit= micron
        # Region Volume Mapping
        result_rvm = RegionVolumeMapping()
        result_rvm.volume = result_volume
        result_rvm.array_data = vol_parcel
        result_rvm.connectivity = result_connectivity
        result_rvm.title = "Volume mouse brain "
        result_rvm.dimensions_labels = ["X", "Y", "Z"]
        # Volume template
        result_template = StructuralMRI()
        result_template.array_data = template
        result_template.weighting = 'T1'
        result_template.volume = result_volume

        connectivity_index = self.store_complete(result_connectivity)
        volume_index = self.store_complete(result_volume)
        rvm_index = self.store_complete(result_rvm)
        template_index = self.store_complete(result_template)

        return [connectivity_index, volume_index, rvm_index, template_index]

    def get_required_memory_size(self, view_model):
        return -1

    def get_required_disk_size(self, view_model):
        return -1


# the method creates a dictionary with information about which experiments need to be downloaded
def dictionary_builder(tvb_mcc, transgenic_line):
    # open up a list of all of the experiments
    all_experiments = tvb_mcc.get_experiments(dataframe=True, cre=transgenic_line)
    # build dict of injection structure id to experiment list
    ist2e = {}
    for eid in all_experiments.index:
        isti = all_experiments.loc[eid]['primary_injection_structure']
        if isti not in ist2e:
            ist2e[isti] = []
        ist2e[isti].append(eid)
    return ist2e


# the method downloads experiments necessary to build the connectivity
def download_an_construct_matrix(tvb_mcc, weighting, ist2e, transgenic_line):
    projmaps = {}
    if weighting == 3:  # download projection energy
        for isti, elist in ist2e.items():
            projmaps[isti] = tvb_mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_energy')
            LOGGER.info('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                        projmaps[isti]['matrix'].shape)
    else:  # download projection density:
        for isti, elist in ist2e.items():
            projmaps[isti] = tvb_mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_density')
            LOGGER.info('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                        projmaps[isti]['matrix'].shape)
        if weighting == 1:  # download injection density
            injdensity = {}
            all_experiments = tvb_mcc.get_experiments(dataframe=True, cre=transgenic_line)
            for exp_id in all_experiments['id']:
                inj_d = tvb_mcc.get_injection_density(exp_id, file_name=None)
                # all the experiments have only an injection sites (only 3 coordinates),
                # thus it is possible to sum the injection matrix
                injdensity[exp_id] = (np.sum(inj_d[0]) / np.count_nonzero(inj_d[0]))
                LOGGER.info('Experiment id', exp_id, ', the total injection density is ', injdensity[exp_id])
            # in this case projmaps will contain PD/ID
            for inj_id in range(len(list(projmaps.values()))):
                index = 0
                for exp_id in list(projmaps.values())[inj_id]['rows']:
                    list(projmaps.values())[inj_id]['matrix'][index] = list(projmaps.values())[inj_id]['matrix'][
                                                                           index] / \
                                                                       injdensity[exp_id]
                    index += 1
    return projmaps


# the method cleans the file projmaps in 4 steps
def pms_cleaner(projmaps):
    def get_structure_id_set(pm):
        return set([c['structure_id'] for c in pm['columns']])

    sis0 = get_structure_id_set(projmaps[502])
    # 1) All the target sites are the same for all the injection sites? If not remove those injection sites
    for inj_id in list(projmaps):
        sis_i = get_structure_id_set(projmaps[inj_id])
        if len(sis0.difference(sis_i)) != 0:
            projmaps.pop(inj_id, None)
    # 2) All the injection sites are also target sites? If not remove those injection sites
    for inj_id in projmaps:
        if inj_id not in sis0:
            del projmaps[inj_id]
    # 3) All the target sites are also injection sites? if not remove those targets from the columns and from the matrix
    if len(sis0) != len(list(projmaps)):
        for inj_id in range(len(list(projmaps.values()))):
            targ_id = -1
            while len(list(projmaps.values())[inj_id]['columns']) != (3 * len(list(projmaps))):
                # there is -3 since for each id-target I have 3 regions since I have 3 hemisphere to consider
                targ_id += 1
                if list(projmaps.values())[inj_id]['columns'][targ_id]['structure_id'] not in list(projmaps):
                    del list(projmaps.values())[inj_id]['columns'][targ_id]
                    list(projmaps.values())[inj_id]['matrix'] = np.delete(list(projmaps.values())[inj_id]['matrix'],
                                                                          targ_id, 1)
                    targ_id = -1
    # 4) Exclude the areas that have NaN values (in all the experiments)
    nan_id = {}
    for inj_id in projmaps.keys():
        mat = projmaps[inj_id]['matrix']
        for targ_id in range(mat.shape[1]):
            if all([np.isnan(mat[exp, targ_id]) for exp in range(mat.shape[0])]):
                if inj_id not in list(nan_id):
                    nan_id[inj_id] = []
                nan_id[inj_id].append(projmaps[inj_id]['columns'][targ_id]['structure_id'])
    while bool(nan_id):
        remove = []
        nan_inj_max = 0
        while list(nan_id)[0] != nan_inj_max:
            len_max = 0
            for inj_id in list(nan_id):
                if len(nan_id[inj_id]) > len_max:
                    nan_inj_max = inj_id
                    len_max = len(nan_id[inj_id])
            if list(nan_id)[0] != nan_inj_max:
                nan_id.pop(nan_inj_max)
                remove.append(nan_inj_max)
        if len(remove) == 0:
            for inj_id in nan_id:
                for target_id in nan_id[inj_id]:
                    if target_id not in remove:
                        remove.append(target_id)
        for rem in remove:
            if rem in list(projmaps):
                projmaps.pop(rem)
            # Remove Nan areas from targe list (columns+matrix)
            for inj_id in range(len(list(projmaps))):
                targ_id = -1
                previous_size = len(list(projmaps.values())[inj_id]['columns'])
                while len(list(projmaps.values())[inj_id]['columns']) != (previous_size - 3):  # 3 hemispheres
                    targ_id += 1
                    column = list(projmaps.values())[inj_id]['columns'][targ_id]
                    if column['structure_id'] == rem:
                        del list(projmaps.values())[inj_id]['columns'][targ_id]
                        list(projmaps.values())[inj_id]['matrix'] = np.delete(list(projmaps.values())[inj_id]['matrix'],
                                                                              targ_id, 1)
                        targ_id = -1
                        # evaluate if there are still Nan values in the matrices
        nan_id = {}
        for inj_id in projmaps:
            mat = projmaps[inj_id]['matrix']
            for targ_id in range(mat.shape[1]):
                if all([np.isnan(mat[exp, targ_id]) for exp in range(mat.shape[0])]):
                    if inj_id not in list(nan_id):
                        nan_id[inj_id] = []
                    nan_id[inj_id].append(projmaps[inj_id]['columns'][targ_id]['structure_id'])

    return projmaps


def areas_volume_threshold(tvb_mcc, projmaps, vol_thresh, resolution):
    """
    the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
    """
    threshold = vol_thresh / (resolution ** 3)
    id_ok = []
    for ID in projmaps:
        mask, _ = tvb_mcc.get_structure_mask(ID)
        tot_voxels = (np.count_nonzero(mask)) / 2  # mask contains both left and right hemisphere
        if tot_voxels > threshold:
            id_ok.append(ID)
            # Remove areas that are not in id_ok from the injection list
    for checkid in list(projmaps):
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove areas that are not in id_ok from target list (columns+matrix)
    for inj_id in range(len(list(projmaps.values()))):
        targ_id = -1
        while len(list(projmaps.values())[inj_id]['columns']) != (len(id_ok) * 3):  # I have 3 hemispheres
            targ_id += 1
            if list(projmaps.values())[inj_id]['columns'][targ_id]['structure_id'] not in id_ok:
                del list(projmaps.values())[inj_id]['columns'][targ_id]
                list(projmaps.values())[inj_id]['matrix'] = np.delete(list(projmaps.values())[inj_id]['matrix'],
                                                                      targ_id, 1)
                targ_id = -1
    return projmaps


# the method includes in the dataset for creating the SC only the experiments whose fraction of infected pixels (in the injection site)
# is greater than inj_f_threshold
def infected_threshold(tvb_mcc, projmaps, inj_f_threshold):
    id_ok = []
    for ID in projmaps:
        exp_not_accepted = []
        for exp in projmaps[ID]['rows']:
            inj_info = tvb_mcc.get_structure_unionizes([exp], is_injection=True, structure_ids=[ID],
                                                       include_descendants=True, hemisphere_ids=[2])
            if len(inj_info) == 0:
                exp_not_accepted.append(exp)
            else:
                inj_f = inj_info['sum_projection_pixels'][0] / inj_info['sum_pixels'][0]
                if inj_f < inj_f_threshold:
                    exp_not_accepted.append(exp)
        if len(exp_not_accepted) < len(projmaps[ID]['rows']):
            id_ok.append(ID)
            projmaps[ID]['rows'] = list(set(projmaps[ID]['rows']).difference(set(exp_not_accepted)))
    for checkid in list(projmaps):
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove areas that are not in id_ok from target list (columns+matrix)
    for indexinj in range(len(list(projmaps.values()))):
        indextarg = -1
        while len(list(projmaps.values())[indexinj]['columns']) != (len(id_ok) * 3):  # I have 3 hemispheres
            indextarg += 1
            if list(projmaps.values())[indexinj]['columns'][indextarg]['structure_id'] not in id_ok:
                del list(projmaps.values())[indexinj]['columns'][indextarg]
                list(projmaps.values())[indexinj]['matrix'] = np.delete(list(projmaps.values())[indexinj]['matrix'],
                                                                        indextarg, 1)
                indextarg = -1
    return projmaps


def create_file_order(projmaps, structure_tree):
    """
    the method creates file order and keyord that will be the link between the structural conn
    order and the id key in the Allen database
    """
    order = {}
    for index in range(len(projmaps)):
        target_id = list(projmaps.values())[0]['columns'][index]['structure_id']
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']] = [target_id]
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['name'])
    key_ord = list(order)
    key_ord.sort()
    return order, key_ord


# the method builds the Structural Connectivity (SC) matrix
def construct_structural_conn(projmaps, order, key_ord):
    len_right = len(list(projmaps))
    structural_conn = np.zeros((len_right, 2 * len_right), dtype=float)
    row = -1
    for graph_ord_inj in key_ord:
        row += 1
        inj_id = order[graph_ord_inj][0]
        target = projmaps[inj_id]['columns']
        matrix = projmaps[inj_id]['matrix']
        # average on the experiments (NB: if there are NaN values not average!)
        if np.isnan(np.sum(matrix)):
            matrix_temp = np.zeros((matrix.shape[1], 1), dtype=float)
            for i in range(matrix.shape[1]):
                if np.isnan(sum(matrix[:, i])):
                    occ = 0
                    for jj in range(matrix.shape[0]):
                        if matrix[jj, i] == matrix[jj, i]:  # since nan!=nan
                            occ += 1
                            matrix_temp[i, 0] = matrix_temp[i, 0] + matrix[jj, i]
                    matrix_temp[i, 0] = matrix_temp[i, 0] / occ
                else:
                    matrix_temp[i, 0] = sum(matrix[:, i]) / matrix.shape[0]
            matrix = matrix_temp
        else:
            matrix = (np.array([sum(matrix[:, i]) for i in range(matrix.shape[1])]) / (matrix.shape[0]))
        # order the target
        col = -1
        for graph_ord_targ in key_ord:
            col += 1
            targ_id = order[graph_ord_targ][0]
            for index in range(len(target)):
                if target[index]['structure_id'] == targ_id:
                    if target[index]['hemisphere_id'] == 2:
                        structural_conn[row, col] = matrix[index]
                    if target[index]['hemisphere_id'] == 1:
                        structural_conn[row, col + len_right] = matrix[index]
    # save the complete matrix (both left and right inj):
    first_quarter = structural_conn[:, :(structural_conn.shape[1] // 2)]
    second_quarter = structural_conn[:, (structural_conn.shape[1] // 2):]
    sc_down = np.concatenate((second_quarter, first_quarter), axis=1)
    structural_conn = np.concatenate((structural_conn, sc_down), axis=0)
    structural_conn = structural_conn / (np.amax(structural_conn))  # normalize the matrix
    return structural_conn.T  # transpose because TVB convention requires SC[target, source]!


# the method returns the centres of the brain areas in the selected parcellation
def construct_centres(tvb_mcc, order, key_ord):
    centres = np.zeros((len(key_ord) * 2, 3), dtype=float)
    names = []
    row = -1
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        coord = [0, 0, 0]
        mask, _ = tvb_mcc.get_structure_mask(node_id)
        mask = rotate_reference(mask)
        mask_r = mask[:mask.shape[0] // 2, :, :]
        xyz = np.where(mask_r)
        if xyz[0].shape[0] > 0:  # Check if the area is in the annotation volume
            coord[0] = np.mean(xyz[0])
            coord[1] = np.mean(xyz[1])
            coord[2] = np.mean(xyz[2])
        row += 1
        centres[row, :] = coord
        coord[0] = (mask.shape[0]) - coord[0]
        centres[row + len(key_ord), :] = coord
        n = order[graph_ord_inj][1]
        right = 'Right '
        right += n
        right = str(right)
        names.append(right)
    for graph_ord_inj in key_ord:
        n = order[graph_ord_inj][1]
        left = 'Left '
        left += n
        left = str(left)
        names.append(left)
    return centres, names


# the method returns the tract lengths between the brain areas in the selected parcellation
def construct_tract_lengths(centres):
    len_right = len(centres) // 2
    tracts = np.zeros((len_right, len(centres)), dtype=float)
    for inj in range(len_right):
        center_inj = centres[inj]
        for targ in range(len_right):
            targ_r = centres[targ]
            targ_l = centres[targ + len_right]
            tracts[inj, targ] = np.sqrt(
                (center_inj[0] - targ_r[0]) ** 2 + (center_inj[1] - targ_r[1]) ** 2 + (center_inj[2] - targ_r[2]) ** 2)
            tracts[inj, targ + len_right] = np.sqrt(
                (center_inj[0] - targ_l[0]) ** 2 + (center_inj[1] - targ_l[1]) ** 2 + (center_inj[2] - targ_l[2]) ** 2)
    # Save the complete matrix (both left and right inj):
    first_quarter = tracts[:, :(tracts.shape[1] // 2)]
    second_quarter = tracts[:, (tracts.shape[1] // 2):]
    tracts_down = np.concatenate((second_quarter, first_quarter), axis=1)
    tracts = np.concatenate((tracts, tracts_down), axis=0)
    return tracts.T  # transpose because TVB convention requires SC[target, source]!


# the method associated the parent and the grandparents to the child in the selected parcellation with the biggest vol
# Since the parcellation is reduced some areas are in the annotation volume but not in the parcellation,
# so it is possible to plot also those areas with following trick:
# If an area that is not in the parcellation is brother of an area that is in the parcellation (same parent),
# the areas not in the parcellation will be plotted in the vol with the
# same vec_indexed of the area in the parcellation.
# In order to have an univocal relation, since some areas in the parcellation have some parent
# for each parent it will be link the child with the biggest volume in the parcellation
# the same is done for the grandparents
def parents_and_grandparents_finder(tvb_mcc, order, key_ord, structure_tree):
    parents = []  # Here it will be the id of the parents of the areas in the parcellation
    grandparents = []  # Here it will be the id of the grandparents of the areas in the parcellation
    vol_areas = []  # Here it will be the volume of the areas in the parcellation
    vec_index = []  # Here it will be the index of the vector of the areas in the parcellation
    index = 0
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        parents.append(structure_tree.get_structures_by_id([node_id])[0]['structure_id_path'][-2])
        grandparents.append(structure_tree.get_structures_by_id([node_id])[0]['structure_id_path'][-3])
        vec_index.append(index)
        index += 1
        mask, _ = tvb_mcc.get_structure_mask(node_id)
        tot_voxels = np.count_nonzero(mask)
        vol_areas.append(tot_voxels)
    # I will order parents, grandparents, vec_index according to the volume of the areas
    parents = [parents for (vv, parents) in sorted(zip(vol_areas, parents))]
    grandparents = [grandparents for (vv, grandparents) in sorted(zip(vol_areas, grandparents))]
    vec_index = [iid for (vv, iid) in sorted(zip(vol_areas, vec_index))]
    k = len(parents)
    unique_parents = {}  # Unique parents will be a dictionary with keys the parent id and as value the index vec
    # of the region in parcellation which has that parent id
    for p in reversed(parents):
        k -= 1
        if p not in list(unique_parents):
            unique_parents[p] = vec_index[k]
    k = len(grandparents)
    unique_gradparents = {}  # Unique parents will be a dictionary with keys the parent id and as value the index vec
    # of the region in my parcellation that has that parent id
    for p in reversed(grandparents):
        k -= 1
        if np.isnan(p) == 0:
            if p not in list(unique_gradparents):
                unique_gradparents[p] = vec_index[k]
    return unique_parents, unique_gradparents


def mouse_brain_visualizer(vol, order, key_ord, unique_parents, unique_grandparents, structure_tree, projmaps):
    """
    the method returns a volume indexed between 0 and N-1, with N=tot brain areas in the parcellation.
    -1=background and areas that are not in the parcellation
    """
    tot_areas = len(key_ord) * 2
    indexed_vec = np.arange(tot_areas).reshape(tot_areas, )
    # vec indexed between 0 and (N-1), with N=total number of area in the parcellation
    indexed_vec = indexed_vec + 1  # vec indexed between 1 and N
    indexed_vec = indexed_vec * (10 ** (-(1 + int(np.log10(tot_areas)))))
    # vec indexed between 0 and 0,N (now all the entries of vec_indexed are < 1 in order to not create confusion
    # with the entry of Vol (always greater than 1)
    vol_r = vol[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)
    index_vec = 0  # this is the index of the vector
    left = len(indexed_vec) // 2
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        if node_id in vol_r:  # check if the area is in the annotation volume
            vol_r[vol_r == node_id] = indexed_vec[index_vec]
            vol_l[vol_l == node_id] = indexed_vec[index_vec + left]
        child = []
        for ii in range(len(structure_tree.children([node_id])[0])):
            child.append(structure_tree.children([node_id])[0][ii]['id'])
        while len(child) != 0:
            if (child[0] in vol_r) and (child[0] not in list(projmaps)):
                vol_r[vol_r == child[0]] = indexed_vec[index_vec]
                vol_l[vol_l == child[0]] = indexed_vec[index_vec + left]
            child.remove(child[0])
        index_vec += 1  # index of vector
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    # Since the parcellation is reduced some areas are in the annotation volume but not in the parcellation,
    # so it is possible to plot also those areas with trick explained in ParentsAndGrandPArentsFinder
    # Parents:
    bool_idx = (vol_parcel > np.amax(indexed_vec))
    # Find the elements of vol_parcel that are yet not associated to a value of the indexed_vec in the parcellation
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)

    for node_id in not_assigned:
        node_id = int(node_id)
        if structure_tree.get_structures_by_id([node_id])[0] is not None:
            ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        else:
            ancestor = []
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in list(unique_parents):
                vol_r[vol_r == node_id] = indexed_vec[unique_parents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_parents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    # Grand parents:
    bool_idx = (vol_parcel > np.amax(indexed_vec))
    # Find the elements of vol_parcel that are yet not associated to a value of the indexed_vec in the parcellation
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)
    for node_id in not_assigned:
        node_id = int(node_id)
        if structure_tree.get_structures_by_id([node_id])[0] is not None:
            ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        else:
            ancestor = []
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in list(unique_grandparents):
                vol_r[vol_r == node_id] = indexed_vec[unique_grandparents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_grandparents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    vol_parcel[vol_parcel >= 1] = 0  # set all the areas not in the parcellation to 0 since the background is zero
    vol_parcel = vol_parcel * (10 ** (1 + int(np.log10(tot_areas))))  # return to indexed between
    # 1 and N (with N=tot number of areas in the parcellation)
    vol_parcel = vol_parcel - 1  # with this operation background and areas not in parcellation will be -1
    # and all the others with the indexed between 0 and N-1
    vol_parcel = np.round(vol_parcel)
    vol_parcel = rotate_reference(vol_parcel)
    return vol_parcel


# the method rotate the Allen 3D (x1,y1,z1) reference in the TVB 3D reference (x2,y2,z2).
# the relation between the different reference system is: x1=z2, y1=x2, z1=y2
def rotate_reference(allen):
    # first rotation in order to obtain: x1=x2, y1=z2, z1=y2
    vol_trans = np.zeros((allen.shape[0], allen.shape[2], allen.shape[1]), dtype=int)
    for x in range(allen.shape[0]):
        vol_trans[x, :, :] = (allen[x, :, :][::-1]).transpose()

    # second rotation in order to obtain: x1=z2, y1=x1, z1=y2
    allen_rotate = np.zeros((allen.shape[2], allen.shape[0], allen.shape[1]), dtype=int)
    for y in range(allen.shape[1]):
        allen_rotate[:, :, y] = (vol_trans[:, :, y]).transpose()
    return allen_rotate
