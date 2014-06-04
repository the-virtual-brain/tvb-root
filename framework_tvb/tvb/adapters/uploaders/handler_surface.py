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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import os
import shutil
import numpy
from scipy import sparse
from zipfile import ZipFile, ZIP_DEFLATED
from tempfile import gettempdir
from cfflib import CData
from tvb.core.adapters.exceptions import ParseException
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.utils import get_unique_file_name
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.gifti.util import GiftiDataType, GiftiIntentCode
from tvb.adapters.uploaders.gifti.gifti import GiftiNVPairs, GiftiMetaData, saveImage, GiftiDataArray, GiftiImage
from tvb.adapters.uploaders.helper_handler import get_uids_dict, get_gifty_file_name
from tvb.adapters.uploaders import constants
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes import surfaces, projections


LOG = get_logger("handle_surface")
NUMPY_TEMP_FOLDER = os.path.join(cfg.TVB_STORAGE, "NUMPY_TMP")

#
# Surface <-> GIFTI
# 


def _create_gifty_array(input_data_list, intent, datatype=GiftiDataType.NIFTI_TYPE_FLOAT32):
    """
    From a input list of points, the intent of this list and the number of points(used for dimensionality)
    create a GIFTI array.
    """
    gii_array = GiftiDataArray()
    gii_array.data = input_data_list
    gii_array.dims.extend(input_data_list.shape)
    gii_array.datatype = datatype
    gii_array.intent = intent
    return gii_array


def _create_gifti_nv_pairs(name, value):
    """
    Creates a GiftiNVPairs with the specified data.
    """
    gifti_pair = GiftiNVPairs()
    gifti_pair.name = name
    gifti_pair.value = value

    return gifti_pair


def _create_c_data(file_prefix, src, metadata_dict, fileformat='Other'):
    """
    Creates a CData object from the given parameters
    """
    if not src:
        return None

    if fileformat == 'NumPy':
        numpy_data = numpy.loadtxt(src)
        if not os.path.exists(NUMPY_TEMP_FOLDER):
            os.makedirs(NUMPY_TEMP_FOLDER)
        temp_file, uq_name = get_unique_file_name(NUMPY_TEMP_FOLDER, file_prefix + ".npy")
        numpy.save(temp_file, numpy_data)
        data = CData(uq_name, temp_file, fileformat='Other')
    else:
        data = CData(file_prefix, src, fileformat=fileformat)

    data.update_metadata(metadata_dict)
    return data


def surface2gifti(surface_data, project_id):
    """ 
    Convert internal SurfaceDataType into GiftiImage. Data tuple will contain
    a Surface Data in the first position and a dictionary 
    with additional information.
    """
    uids_dict = get_uids_dict(surface_data)
    gii_surface_uid = _create_gifti_nv_pairs(constants.KEY_UID, uids_dict[constants.KEY_SURFACE_UID])
    gii_surface_class = _create_gifti_nv_pairs(constants.SURFACE_CLASS, constants.CLASS_SURFACE)
    gii_zero_based = _create_gifti_nv_pairs(constants.PARAM_ZERO_BASED, surface_data.zero_based_triangles)
    gii_surface_type = _create_gifti_nv_pairs(constants.SURFACE_TYPE, surface_data.surface_type)

    gii_meta_object = GiftiMetaData()
    gii_meta_object.data.append(gii_surface_uid)
    gii_meta_object.data.append(gii_surface_class)
    gii_meta_object.data.append(gii_zero_based)
    gii_meta_object.data.append(gii_surface_type)

    vert_meta = {constants.KEY_ROLE: constants.ROLE_VERTICES,
                 constants.KEY_UID: uids_dict[constants.KEY_SURFACE_UID]}
    vert_path = surface_data.vertices_path
    cdata_vertices = _create_c_data(os.path.split(vert_path)[1], vert_path, vert_meta, 'NumPy')
    norm_meta = {constants.KEY_ROLE: constants.ROLE_NORMALS,
                 constants.KEY_UID: uids_dict[constants.KEY_SURFACE_UID]}
    norm_path = surface_data.vertex_normals_path
    cdata_normals = _create_c_data(os.path.split(norm_path)[1], norm_path, norm_meta, 'NumPy')
    triang_meta = {constants.KEY_ROLE: constants.ROLE_TRIANGLES,
                   constants.KEY_UID: uids_dict[constants.KEY_SURFACE_UID]}
    triang_path = surface_data.triangles_path
    cdata_triangles = _create_c_data(os.path.split(triang_path)[1], triang_path, triang_meta, 'NumPy')
    gii_image = GiftiImage()
    gii_image.numDA = 0
    gii_image.set_metadata(gii_meta_object)

    gii_filename = get_gifty_file_name(project_id, 'tvb_surface.surf.gii')
    saveImage(gii_image, gii_filename)
    return gii_filename, cdata_vertices, cdata_normals, cdata_triangles



def gifti2surface(gifti_image, vertices_array=None, normals_array=None, triangles_array=None, storage_path=''):
    """
    Convert GiftiImage object into our internal Surface DataType
    """
    meta = gifti_image.meta.get_data_as_dict()
    if vertices_array:
        tmpdir = os.path.join(gettempdir(), vertices_array.parent_cfile.get_unique_cff_name())
        LOG.debug("Using temporary folder for surface import: " + tmpdir)
        #### Extract SRC from zipFile to TEMP
        _zipfile = ZipFile(vertices_array.parent_cfile.src, 'r', ZIP_DEFLATED)
        try:
            if vertices_array is not None:
                vertices_array = _zipfile.extract(vertices_array.src, tmpdir)
                vertices_array = numpy.load(vertices_array)
            if normals_array is not None:
                normals_array = _zipfile.extract(normals_array.src, tmpdir)
                normals_array = numpy.load(normals_array)
            if triangles_array is not None:
                triangles_array = _zipfile.extract(triangles_array.src, tmpdir)
                triangles_array = numpy.load(triangles_array)
        except:
            raise RuntimeError('Can not extract %s from Connectome file.' % vertices_array)         
    
    if vertices_array is None:
        for i in xrange(int(gifti_image.numDA)):
            # Intent code is stored in the DataArray structure
            darray = gifti_image.darrays[i]
            if darray.intent == GiftiIntentCode.NIFTI_INTENT_POINTSET:
                vertices_array = darray.data
                continue
            if darray.intent == GiftiIntentCode.NIFTI_INTENT_TRIANGLE:
                triangles_array = darray.data
                continue
            if darray.intent == GiftiIntentCode.NIFTI_INTENT_NORMAL:
                normals_array = darray.data
                continue
    zero_based_val = eval(meta[constants.PARAM_ZERO_BASED])
    
    if meta[constants.KEY_UID] in [surfaces.CORTICAL, 'srf_reg13', 'srf_80k']:
        surface = surfaces.CorticalSurface()
    elif meta[constants.KEY_UID] in [surfaces.OUTER_SKIN, 'outer_skin']:
        surface = surfaces.SkinAir()
    else:
        raise RuntimeError('Can not determine surface type. Abandon import operation.')
        
    surface.storage_path = storage_path
    # Remove mappings from MetaData (or too large exception will be thrown)
    del meta[constants.MAPPINGS_DICT]
    surface.set_metadata(meta) 
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
        
    surface.zero_based_triangles = zero_based_val
    surface.vertices = vertices_array
    surface.vertex_normals = normals_array
    if zero_based_val:
        surface.triangles = numpy.array(triangles_array, dtype=numpy.int32)
    else:
        surface.triangles = numpy.array(triangles_array, dtype=numpy.int32) - 1
    surface.triangle_normals = None
    
    # Now check if the triangles of the surface are valid   
    triangles_min_vertex = numpy.amin(surface.triangles)
    if triangles_min_vertex < 0:
        if triangles_min_vertex == -1 and not zero_based_val:
            raise RuntimeError("Your triangles contains a negative vertex index. May be you have a ZERO based surface.")
        else:
            raise RuntimeError("Your triangles contains a negative vertex index: %d" % triangles_min_vertex)
    
    no_of_vertices = len(surface.vertices)        
    triangles_max_vertex = numpy.amax(surface.triangles)
    if triangles_max_vertex >= no_of_vertices:
        if triangles_max_vertex == no_of_vertices and zero_based_val:
            raise RuntimeError("Your triangles contains an invalid vertex index: %d. "
                               "Maybe your surface is NOT ZERO based." % triangles_max_vertex)
        else:
            raise RuntimeError("Your triangles contains an invalid vertex index: %d." % triangles_max_vertex)
    
    return surface, meta[constants.KEY_UID]


def cdata2local_connectivity(local_connectivity_data, meta, storage_path, expected_length=0):
    """
    From a CData entry in CFF, create LocalConnectivity entity.
    """
    ##### expected_length = cortex.region_mapping.shape[0]
    tmpdir = os.path.join(gettempdir(), local_connectivity_data.parent_cfile.get_unique_cff_name())
    LOG.debug("Using temporary folder for Local Connectivity import: " + tmpdir)
    _zipfile = ZipFile(local_connectivity_data.parent_cfile.src, 'r', ZIP_DEFLATED)
    local_connectivity_path = _zipfile.extract(local_connectivity_data.src, tmpdir)
    
    gid = dao.get_last_data_with_uid(meta[constants.KEY_SURFACE_UID], surfaces.CorticalSurface)
    surface_data = ABCAdapter.load_entity_by_gid(gid)
    
    local_connectivity = surfaces.LocalConnectivity()
    local_connectivity.storage_path = storage_path 
    local_connectivity_data = ABCUploader.read_matlab_data(local_connectivity_path, constants.DATA_NAME_LOCAL_CONN)
    
    if local_connectivity_data.shape[0] < expected_length:
        padding = sparse.csc_matrix((local_connectivity_data.shape[0],
                                    expected_length - local_connectivity_data.shape[0]))
        local_connectivity_data = sparse.hstack([local_connectivity_data, padding])
            
        padding = sparse.csc_matrix((expected_length - local_connectivity_data.shape[0],
                                     local_connectivity_data.shape[1]))
        local_connectivity_data = sparse.vstack([local_connectivity_data, padding])
    
    local_connectivity.equation = None
    local_connectivity.matrix = local_connectivity_data        
    local_connectivity.surface = surface_data
    
    uid = meta[constants.KEY_UID] if constants.KEY_UID in meta else None
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    return local_connectivity, uid


def cdata2region_mapping(region_mapping_data, meta, storage_path):
    """
    From a CData entry in CFF, create RegionMapping entity.
    """
    tmpdir = os.path.join(gettempdir(), region_mapping_data.parent_cfile.get_unique_cff_name())
    LOG.debug("Using temporary folder for Region_Mapping import: " + tmpdir)
    _zipfile = ZipFile(region_mapping_data.parent_cfile.src, 'r', ZIP_DEFLATED)
    region_mapping_path = _zipfile.extract(region_mapping_data.src, tmpdir)
    
    gid = dao.get_last_data_with_uid(meta[constants.KEY_SURFACE_UID], surfaces.CorticalSurface)
    surface_data = ABCAdapter.load_entity_by_gid(gid)
    
    gid = dao.get_last_data_with_uid(meta[constants.KEY_CONNECTIVITY_UID], Connectivity)
    connectivity = ABCAdapter.load_entity_by_gid(gid)
    
    region_mapping = surfaces.RegionMapping(storage_path=storage_path)
    region_mapping.array_data = ABCUploader.read_list_data(region_mapping_path, dtype=numpy.int32)
    region_mapping.connectivity = connectivity
    region_mapping.surface = surface_data
    uid = meta[constants.KEY_UID] if constants.KEY_UID in meta else None
    
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    return region_mapping, uid


def cdata2eeg_mapping(eeg_mapping_data, meta, storage_path, expected_shape=0):
    """
    Currently not used
    """
    tmpdir = os.path.join(gettempdir(), eeg_mapping_data.parent_cfile.get_unique_cff_name())
    LOG.debug("Using temporary folder for EEG_Mapping import: " + tmpdir)
    _zipfile = ZipFile(eeg_mapping_data.parent_cfile.src, 'r', ZIP_DEFLATED)
    eeg_projection_path = _zipfile.extract(eeg_mapping_data.src, tmpdir)
    eeg_projection_data = ABCUploader.read_matlab_data(eeg_projection_path, constants.DATA_NAME_PROJECTION)
    if eeg_projection_data.shape[1] < expected_shape:
        padding = numpy.zeros((eeg_projection_data.shape[0], expected_shape - eeg_projection_data.shape[1]))
        eeg_projection_data = numpy.hstack((eeg_projection_data, padding))
        
    gid = dao.get_last_data_with_uid(meta[constants.KEY_SURFACE_UID], surfaces.CorticalSurface)
    surface_data = ABCAdapter.load_entity_by_gid(gid)
    
    projection_matrix = projections.ProjectionSurfaceEEG(storage_path=storage_path)
    projection_matrix.projection_data = eeg_projection_data
    projection_matrix.sources = surface_data
    projection_matrix.sensors = None
    ### TODO if we decide to use this method, we will need to find a manner to fill the sensors.
    return projection_matrix


def create_surface_of_type(surface_type):
    """
    :param surface_type: One of the constants describing the type of the surface in the ui
    :return: A tvb Surface
    """
    if surface_type == constants.OPTION_SURFACE_SKINAIR:
        return surfaces.SkinAir()
    elif surface_type.startswith(constants.OPTION_SURFACE_CORTEX):
        return surfaces.CorticalSurface()
    elif surface_type == constants.OPTION_SURFACE_FACE:
        return surfaces.FaceSurface()
    else:
        raise ParseException("Could not determine type of the surface")
    

def center_vertices(vertices):
    """
    Centres the vertices using means along axes.
    :param vertices: a numpy array of shape (n, 3)
    :returns: the centered array
    """
    return vertices - numpy.mean(vertices, axis=0).reshape((1, 3))
