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
The adapters in this module create new connectivities from the Allen Institute
data using their SDK.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import os.path
import numpy as np
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.entities.storage import dao
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.volumes import Volume
from tvb.datatypes.structural import StructuralMRI
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


LOGGER = get_logger(__name__)


class AllenConnectomeBuilder(ABCAsynchronous):
    """Handler for uploading a mouse connectivity from Allen dataset using AllenSDK."""

    _ui_name = "Allen connectivity builder"
    _ui_description = "Import mouse connectivity from Allen database (tracer experiments)"

    # TRANSGENIC_OPTIONS = [
    #    {'name': 'No', 'value': 'False'},
    #    {'name': 'Yes', 'value': 'True'}
    # ]

    RESOLUTION_OPTIONS = [
        {'name': '25', 'value': '25'},
        {'name': '50', 'value': '50'},
        {'name': '100', 'value': '100'}
    ]

    WEIGHTS_OPTIONS = [
        {'name': '(projection density)/(injection density)', 'value': '1'},
        {'name': 'projection density', 'value': '2'},
        {'name': 'projection energy', 'value': '3'},
    ]

    def get_input_tree(self):
        return [  # {'name': 'TransgenicLine', 'type': 'select', 'label': 'Transgenic line :',
            # 'required': True, 'options': self.TRANSGENIC_OPTIONS, 'default': 'False'},

            {'name': 'resolution', 'type': 'select',
             'label': 'Spatial resolution (micron)',
             'description': 'Resolution of the data that you want to download to construct the volume '
                            'and the connectivity (micron) :',
             'required': True, 'options': self.RESOLUTION_OPTIONS, 'default': '100'},

            {'name': 'weighting', 'type': 'select', 'label': 'Definition of the weights of the connectivity :',
             'required': True, 'options': self.WEIGHTS_OPTIONS, 'default': '1'},

            {'name': 'inf_vox_thresh', 'type': 'float',
             'label': 'Min infected voxels',
             'description': 'To build the volume and the connectivity select only the areas where there is at '
                            'least one infection experiment that had infected more than ... voxels in that areas.',
             'required': True, 'default': '50'},

            {'name': 'vol_thresh', 'type': 'float',
             'label': 'Min volume',
             'description': 'To build the volume and the connectivity select only the areas that have a '
                            'volume greater than (micron^3): ',
             'required': True, 'default': '1000000000'}]

    def get_output(self):
        return [Connectivity, Volume, RegionVolumeMapping, StructuralMRI]


    def launch(self, resolution, weighting, inf_vox_thresh, vol_thresh):
        resolution = int(resolution)
        weighting = int(weighting)
        inf_vox_thresh = float(inf_vox_thresh)
        vol_thresh = float(vol_thresh)

        project = dao.get_project_by_id(self.current_project_id)
        manifest_file = self.file_handler.get_allen_mouse_cache_folder(project.name)
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

        # grab the StructureTree instance
        structure_tree = cache.get_structure_tree()

        # rotate template in the TVB 3D reference:
        template = rotate_reference(template)

        # the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
        projmaps = areas_volume_threshold(cache, projmaps, vol_thresh, resolution)

        # the method includes in the parcellation only brain regions where at least one injection experiment
        # had infected more than N voxel (where N is inf_vox_thresh)
        projmaps = areas_voxel_threshold(cache, projmaps, inf_vox_thresh, vol, structure_tree)

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
        result_connectivity = Connectivity(storage_path=self.storage_path)
        result_connectivity.centres = centres
        result_connectivity.region_labels = names
        result_connectivity.weights = structural_conn
        result_connectivity.tract_lengths = tract_lengths
        # Volume
        result_volume = Volume(storage_path=self.storage_path)
        result_volume.origin = [[0.0, 0.0, 0.0]]
        result_volume.voxel_size = [resolution, resolution, resolution]
        # result_volume.voxel_unit= micron
        # Region Volume Mapping
        result_rvm = RegionVolumeMapping(storage_path=self.storage_path)
        result_rvm.volume = result_volume
        result_rvm.array_data = vol_parcel
        result_rvm.connectivity = result_connectivity
        result_rvm.title = "Volume mouse brain "
        result_rvm.dimensions_labels = ["X", "Y", "Z"]
        # Volume template
        result_template = StructuralMRI(storage_path=self.storage_path)
        result_template.array_data = template
        result_template.weighting = 'T1'
        result_template.volume = result_volume
        return [result_connectivity, result_volume, result_rvm, result_template]

    def get_required_memory_size(self, **kwargs):
        return -1

    def get_required_disk_size(self, **kwargs):
        return -1


# the method creates a dictionary with information about which experiments need to be downloaded
def dictionary_builder(tvb_mcc, transgenic_line):
    # open up a list of all of the experiments
    all_experiments = tvb_mcc.get_experiments(dataframe=True, cre=transgenic_line)
    # build dict of injection structure id to experiment list
    ist2e = {}
    for eid in all_experiments.index:
        for ist in all_experiments.ix[eid]['injection-structures']:
            isti = ist['id']
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
                projection_structure_ids=ist2e.keys(),  # summary_structure_ids,
                parameter='projection_energy')
            LOGGER.info('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                        projmaps[isti]['matrix'].shape)
    else:  # download projection density:
        for isti, elist in ist2e.items():
            projmaps[isti] = tvb_mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=ist2e.keys(),  # summary_structure_ids,
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
            for inj_id in range(len(projmaps.values())):
                index = 0
                for exp_id in projmaps.values()[inj_id]['rows']:
                    projmaps.values()[inj_id]['matrix'][index] = projmaps.values()[inj_id]['matrix'][index] / \
                                                                 injdensity[exp_id]
                    index += 1
    return projmaps


# the method cleans the file projmaps in 4 steps
def pms_cleaner(projmaps):
    def get_structure_id_set(pm):
        return set([c['structure_id'] for c in pm['columns']])

    sis0 = get_structure_id_set(projmaps[515])
    # 1) All the target sites are the same for all the injection sites? If not remove those injection sites
    for inj_id in projmaps.keys():
        sis_i = get_structure_id_set(projmaps[inj_id])
        if len(sis0.difference(sis_i)) != 0:
            projmaps.pop(inj_id, None)
    # 2) All the injection sites are also target sites? If not remove those injection sites
    for inj_id in projmaps.keys():
        if inj_id not in sis0:
            del projmaps[inj_id]
    # 3) All the target sites are also injection sites? if not remove those targets from the columns and from the matrix
    if len(sis0) != len(projmaps.keys()):
        for inj_id in range(len(projmaps.values())):
            targ_id = -1
            while len(projmaps.values()[inj_id]['columns']) != (3 * len(projmaps.keys())):
                # there is -3 since for each id-target I have 3 regions since I have 3 hemisphere to consider
                targ_id += 1
                if projmaps.values()[inj_id]['columns'][targ_id]['structure_id'] not in projmaps.keys():
                    del projmaps.values()[inj_id]['columns'][targ_id]
                    projmaps.values()[inj_id]['matrix'] = np.delete(projmaps.values()[inj_id]['matrix'], targ_id, 1)
                    targ_id = -1
    # 4) Exclude the areas that have NaN values (in all the experiments)
    nan_id = {}
    for inj_id in projmaps.keys():
        mat = projmaps[inj_id]['matrix']
        for targ_id in range(mat.shape[1]):
            if all([np.isnan(mat[exp, targ_id]) for exp in range(mat.shape[0])]):
                if inj_id not in nan_id.keys():
                    nan_id[inj_id] = []
                nan_id[inj_id].append(projmaps[inj_id]['columns'][targ_id]['structure_id'])
    while bool(nan_id):
        # I created the while in order to remove less structure as possible, maybe in some case the nan values refer
        # always to the same target structure (a lot of inj structure could have nan value for the same target),
        # in order to prevent to remove all that inj structure I will remove only the one with more target nan values
        # and I will check if the problem is solved in the last part of the while loop
        len_remove = 0
        for i in nan_id.keys():
            if len(nan_id[i]) > len_remove:
                remove = i
                len_remove = len(nan_id)
        nan_id.pop(remove, None)
        projmaps.pop(remove, None)
        # Remove Nan areas from target list (columns+matrix)
        for inj_id in range(len(projmaps.keys())):
            targ_id = -1
            curr_columns_arr = projmaps.values()[inj_id]['columns']
            previous_size = len(curr_columns_arr)
            while len(curr_columns_arr) != (previous_size - 3 * len(nan_id)) and targ_id + 1 < len(curr_columns_arr):
                # 3 hemispheres
                targ_id += 1
                column = curr_columns_arr[targ_id]
                if column and 'structure_id' in column and column['structure_id'] == remove:
                    del curr_columns_arr[targ_id]
                    projmaps.values()[inj_id]['matrix'] = np.delete(projmaps.values()[inj_id]['matrix'], targ_id, 1)
                    targ_id = -1
        # evaluate if there are still Nan values in the matrices
        nan_id = {}
        for inj_id in projmaps.keys():
            mat = projmaps[inj_id]['matrix']
            for targ_id in range(mat.shape[1]):
                if all([np.isnan(mat[exp, targ_id]) for exp in range(mat.shape[0])]):
                    if inj_id not in nan_id.keys():
                        nan_id[inj_id] = []
                    nan_id[inj_id].append(projmaps[inj_id]['columns'][targ_id]['structure_id'])

    return projmaps


def areas_volume_threshold(tvb_mcc, projmaps, vol_thresh, resolution):
    """
    the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
    """
    threshold = vol_thresh / (resolution ** 3)
    id_ok = []
    for ID in projmaps.keys():
        mask, _ = tvb_mcc.get_structure_mask(ID)
        tot_voxels = (np.count_nonzero(mask)) / 2  # mask contains both left and right hemisphere
        if tot_voxels > threshold:
            id_ok.append(ID)
            # Remove areas that are not in id_ok from the injection list
    for checkid in projmaps.keys():
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove areas that are not in id_ok from target list (columns+matrix)
    for inj_id in range(len(projmaps.values())):
        targ_id = -1
        while len(projmaps.values()[inj_id]['columns']) != (len(id_ok) * 3):  # I have 3 hemispheres
            targ_id += 1
            if projmaps.values()[inj_id]['columns'][targ_id]['structure_id'] not in id_ok:
                del projmaps.values()[inj_id]['columns'][targ_id]
                projmaps.values()[inj_id]['matrix'] = np.delete(projmaps.values()[inj_id]['matrix'], targ_id, 1)
                targ_id = -1
    return projmaps


# the method includes in the parcellation only brain regions where at least one injection experiment had infected
# more than N voxel (where N is inf_vox_thresh)
# what can be download is the Injection fraction: fraction of pixels belonging to manually annotated injection site
# (http://help.brain-map.org/display/mouseconnectivity/API)
# when this fraction is greater than inf_vox_thresh that area is included in the parcellation
def areas_voxel_threshold(tvb_mcc, projmaps, inf_vox_thresh, vol, structure_tree):
    id_ok = []
    for ID in projmaps.keys():
        for exp in projmaps[ID]['rows']:
            inj_f, inf_info = tvb_mcc.get_injection_fraction(exp)
            inj_voxels = np.where(inj_f >= 0)  # coordinates of the injected voxels with inj fraction >= 0
            id_infected = vol[inj_voxels]  # id of area located in Vol where injf >= 0
            #  (thus the id of the infected structure)
            myid_infected = np.where(id_infected == ID)
            inj_f_id = 0  # fraction of injected voxels for the ID I am currently examining
            for index in myid_infected[0]:
                inj_f_id += inj_f[inj_voxels[0][index], inj_voxels[1][index], inj_voxels[2][index]]
                if inj_f_id >= inf_vox_thresh:
                    break
            if inj_f_id >= inf_vox_thresh:
                id_ok.append(ID)
                break
            else:  # check for child of that area
                child = []
                for ii in range(len(structure_tree.children([ID])[0])):
                    child.append(structure_tree.children([ID])[0][ii]['id'])
                while len(child) != 0:
                    myid_infected = np.where(id_infected == child[0])
                    if len(myid_infected[0]) != 0:
                        for index in myid_infected[0]:
                            inj_f_id += inj_f[inj_voxels[0][index], inj_voxels[1][index], inj_voxels[2][index]]
                            if inj_f_id >= inf_vox_thresh:
                                break
                        if inj_f_id >= inf_vox_thresh:
                            id_ok.append(ID)
                            break
                    child.remove(child[0])
            if inj_f_id >= inf_vox_thresh:
                break
    # Remove areas that are not in id_ok from the injection list
    for checkid in projmaps.keys():
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove areas that are not in id_ok from target list (columns+matrix)
    for indexinj in range(len(projmaps.values())):
        indextarg = -1
        while len(projmaps.values()[indexinj]['columns']) != (len(id_ok) * 3):  # I have 3 hemispheres
            indextarg += 1
            if projmaps.values()[indexinj]['columns'][indextarg]['structure_id'] not in id_ok:
                del projmaps.values()[indexinj]['columns'][indextarg]
                projmaps.values()[indexinj]['matrix'] = np.delete(projmaps.values()[indexinj]['matrix'], indextarg, 1)
                indextarg = -1
    return projmaps


def create_file_order(projmaps, structure_tree):
    """
    the method creates file order and keyord that will be the link between the structural conn
    order and the id key in the Allen database
    """
    order = {}
    for index in range(len(projmaps)):
        target_id = projmaps.values()[0]['columns'][index]['structure_id']
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']] = [target_id]
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['name'])
    key_ord = order.keys()
    key_ord.sort()
    return order, key_ord


# the method builds the Structural Connectivity (SC) matrix
def construct_structural_conn(projmaps, order, key_ord):
    len_right = len(projmaps.keys())
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
    first_quarter = structural_conn[:, :(structural_conn.shape[1] / 2)]
    second_quarter = structural_conn[:, (structural_conn.shape[1] / 2):]
    sc_down = np.concatenate((second_quarter, first_quarter), axis=1)
    structural_conn = np.concatenate((structural_conn, sc_down), axis=0)
    structural_conn = structural_conn / (np.amax(structural_conn))  # normalize the matrix
    return structural_conn


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
        mask_r = mask[:mask.shape[0] / 2, :, :]
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
    len_right = len(centres) / 2
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
    first_quarter = tracts[:, :(tracts.shape[1] / 2)]
    second_quarter = tracts[:, (tracts.shape[1] / 2):]
    tracts_down = np.concatenate((second_quarter, first_quarter), axis=1)
    tracts = np.concatenate((tracts, tracts_down), axis=0)
    return tracts


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
        if p not in unique_parents.keys():
            unique_parents[p] = vec_index[k]
    k = len(grandparents)
    unique_gradparents = {}  # Unique parents will be a dictionary with keys the parent id and as value the index vec
    # of the region in my parcellation that has that parent id
    for p in reversed(grandparents):
        k -= 1
        if np.isnan(p) == 0:
            if p not in unique_gradparents.keys():
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
    vol_r = vol[:, :, :(vol.shape[2] / 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol[:, :, (vol.shape[2] / 2):]
    vol_l = vol_l.astype(np.float64)
    index_vec = 0  # this is the index of the vector
    left = len(indexed_vec) / 2
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        if node_id in vol_r:  # check if the area is in the annotation volume
            vol_r[vol_r == node_id] = indexed_vec[index_vec]
            vol_l[vol_l == node_id] = indexed_vec[index_vec + left]
        child = []
        for ii in range(len(structure_tree.children([node_id])[0])):
            child.append(structure_tree.children([node_id])[0][ii]['id'])
        while len(child) != 0:
            if (child[0] in vol_r) and (child[0] not in projmaps.keys()):
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
    vol_r = vol_parcel[:, :, :(vol.shape[2] / 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] / 2):]
    vol_l = vol_l.astype(np.float64)
    for node_id in not_assigned:
        node_id = int(node_id)
        ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in unique_parents.keys():
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
    vol_r = vol_parcel[:, :, :(vol.shape[2] / 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] / 2):]
    vol_l = vol_l.astype(np.float64)
    for node_id in not_assigned:
        node_id = int(node_id)
        ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in unique_grandparents.keys():
                vol_r[vol_r == node_id] = indexed_vec[unique_parents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_parents[pp] + left]
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
    vol_trans = np.zeros((allen.shape[0], allen.shape[2], allen.shape[1]), dtype=float)
    for x in range(allen.shape[0]):
        vol_trans[x, :, :] = (allen[x, :, :][::-1]).transpose()

    # second rotation in order to obtain: x1=z2, y1=x1, z1=y2
    allen_rotate = np.zeros((allen.shape[2], allen.shape[0], allen.shape[1]), dtype=float)
    for y in range(allen.shape[1]):
        allen_rotate[:, :, y] = (vol_trans[:, :, y]).transpose()
    return allen_rotate
