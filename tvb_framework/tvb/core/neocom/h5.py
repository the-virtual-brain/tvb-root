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

import os
import typing
import uuid

from tvb.basic.neotraits.api import HasTraits
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neocom._h5loader import Loader, DirLoader, TVBLoader, ViewModelLoader, DtLoader
from tvb.core.neocom._registry import Registry
from tvb.core.neotraits.h5 import H5File
from tvb.core.neotraits.view_model import ViewModel

REGISTRY = Registry()


def path_for_stored_index(dt_index_instance):
    # type: (DataType) -> str
    loader = TVBLoader(REGISTRY)
    return loader.path_for_stored_index(dt_index_instance)


def path_for(op_id, h5_file_class, gid, project_name, dt_class=None):
    # type: (int, typing.Type[H5File], object, str, str) -> str
    loader = TVBLoader(REGISTRY)
    return loader.path_for(op_id, h5_file_class, gid, project_name, dt_class)


def path_by_dir(base_dir, h5_file_class, gid, dt_class=None):
    # type: (str, typing.Type[H5File], str, str) -> str
    loader = TVBLoader(REGISTRY)
    return loader.path_by_dir(base_dir, h5_file_class, gid, dt_class)


def h5_file_for_index(dt_index_instance):
    # type: (DataType) -> H5File
    h5_path = path_for_stored_index(dt_index_instance)
    h5_class = REGISTRY.get_h5file_for_index(type(dt_index_instance))
    return h5_class(h5_path)


def index_for_h5_file(source_path):
    # type: (str) -> typing.Type[DataType]
    """"""
    h5_class = H5File.h5_class_from_file(source_path)
    return REGISTRY.get_index_for_h5file(h5_class)


def h5_file_for_gid(data_gid):
    # type: (str) -> H5File
    datatype_index = load_entity_by_gid(data_gid)
    return h5_file_for_index(datatype_index)


def load_from_gid(data_gid):
    # type: (str) -> HasTraits
    datatype_index = load_entity_by_gid(data_gid)
    return load_from_index(datatype_index)


def load_from_index(dt_index):
    # type: (DataType) -> HasTraits
    loader = TVBLoader(REGISTRY)
    return loader.load_from_index(dt_index)


def load(source_path, with_references=False):
    # type: (str, bool) -> HasTraits
    """
    Load a datatype stored in the tvb h5 file found at the given path
    """
    if with_references:
        loader = DirLoader(os.path.dirname(source_path), REGISTRY, with_references)
        return loader.load(fname=os.path.basename(source_path))
    else:
        loader = Loader(REGISTRY)
        return loader.load(source_path)


def load_with_references(source_path):
    # type: (str) -> (HasTraits, GenericAttributes)
    """
    Load a datatype stored in the tvb h5 file found at the given path, but also load linked entities through GID
    """
    loader = TVBLoader(REGISTRY)
    return loader.load_with_references(source_path)


def load_with_links(source_path):
    # type: (str) -> (HasTraits, GenericAttributes)
    """
    Load a datatype stored in the tvb h5 file found at the given path, but also create empty linked entities to hold GID
    """
    loader = TVBLoader(REGISTRY)
    return loader.load_with_links(source_path)


def __store_complete(datatype, storage_path, h5_class, generic_attributes=GenericAttributes()):
    # type: (HasTraits, str, type(H5File), GenericAttributes) -> DataType
    """
    Stores the given HasTraits instance in a h5 file, and fill the Index entity for later storage in DB
    """
    index_class = REGISTRY.get_index_for_datatype(datatype.__class__)
    index_inst = index_class()
    index_inst.fill_from_has_traits(datatype)
    index_inst.fill_from_generic_attributes(generic_attributes)

    with h5_class(storage_path) as f:
        f.store(datatype)
        # Store empty Generic Attributes, in case the file is saved no through ABCAdapter it can still be used
        f.store_generic_attributes(generic_attributes)

    return index_inst


def store_complete_to_dir(datatype, base_dir, generic_attributes=GenericAttributes()):
    h5_class = REGISTRY.get_h5file_for_datatype(datatype.__class__)
    storage_path = path_by_dir(base_dir, h5_class, datatype.gid)

    index_inst = __store_complete(datatype, storage_path, h5_class, generic_attributes)
    return index_inst


def store_complete(datatype, op_id, project_name, generic_attributes=GenericAttributes()):
    h5_class = REGISTRY.get_h5file_for_datatype(datatype.__class__)
    storage_path = path_for(op_id, h5_class, datatype.gid, project_name)

    index_inst = __store_complete(datatype, storage_path, h5_class, generic_attributes)
    return index_inst


def store(datatype, destination, recursive=False):
    # type: (HasTraits, str, bool) -> None
    """
    Stores the given datatype in a tvb h5 file at the given path
    """
    if recursive:
        loader = DirLoader(os.path.dirname(destination), REGISTRY, recursive)
        loader.store(datatype, os.path.basename(destination))
    else:
        loader = Loader(REGISTRY)
        loader.store(datatype, destination)


def load_from_dir(base_dir, gid, recursive=False):
    # type: (str, typing.Union[str, uuid.UUID], bool) -> HasTraits
    """
    Loads a datatype with the requested gid from the given directory.
    The datatype should have been written with store_to_dir
    The name and location of the file is chosen for you.
    :param base_dir:  The h5 storage directory
    :param gid: the gid of the to be loaded datatype
    :param recursive: if datatypes contained in this datatype should be loaded as well
    """
    loader = DirLoader(base_dir, REGISTRY, recursive)
    return loader.load(gid)


def load_with_links_from_dir(base_dir, gid):
    # type: (str, typing.Union[uuid.UUID, str]) -> HasTraits
    dir_loader = DirLoader(base_dir, REGISTRY, False)
    fname = dir_loader.find_file_by_gid(gid)
    tvb_loader = TVBLoader(REGISTRY)
    return tvb_loader.load_with_links(fname)


def load_with_references_from_dir(base_dir, gid):
    # type: (str, typing.Union[uuid.UUID, str]) -> HasTraits
    dir_loader = DirLoader(base_dir, REGISTRY, False)
    fname = dir_loader.find_file_by_gid(gid)
    tvb_loader = TVBLoader(REGISTRY)

    def load_ht_function(sub_gid, traited_attr):
        return dir_loader.load(sub_gid)

    return tvb_loader.load_complete_by_function(fname, load_ht_function)


def store_to_dir(datatype, base_dir, recursive=False):
    # type: (HasTraits, str, bool) -> str
    """
    Stores the given datatype in the given directory.
    The name and location of the stored file(s) is chosen for you by this function.
    If recursive is true than datatypes referenced by this one are stored as well.
    """
    loader = DirLoader(base_dir, REGISTRY, recursive)
    return loader.store(datatype)


def determine_filepath(gid, base_dir):
    """
    Find the file path containing a datatype with the given GID within the directory specified by base_dir
    """
    dir_loader = DirLoader(base_dir, REGISTRY, False)
    fname = dir_loader.find_file_by_gid(gid)
    return fname


def store_ht(ht, base_dir):
    # type: (HasTraits, str)-> str
    """
    Completely store any HasTraits object to the directory specified by base_dir
    """
    loader = DtLoader(base_dir, REGISTRY)
    return loader.store(ht)


def load_ht(gid, base_dir):
    # type: (typing.Union[uuid.UUID, str], str)-> str
    """
    Completely load any HasTraits object with the gid specified from the base_dir directory
    """
    loader = DtLoader(base_dir, REGISTRY)
    return loader.load(gid)


def store_view_model(view_model, base_dir):
    # type: (ViewModel, str) -> str
    """
    Completely store any ViewModel object to the directory specified by base_dir.
    """
    vm_loader = ViewModelLoader(base_dir)
    return vm_loader.store(view_model)


def load_view_model(gid, base_dir):
    # type: (typing.Union[uuid.UUID, str], str) -> ViewModel
    """
    Load a ViewModel object by reading the H5 file with the given GID, from the directory specified by base_dir.
    """
    vm_loader = ViewModelLoader(base_dir)
    return vm_loader.load(gid)


def load_view_model_from_file(filepath):
    # type: (str) -> ViewModel
    """
    Load a ViewModel object by reading the H5 file specified by filepath.
    """
    base_dir = os.path.dirname(filepath)
    fname = os.path.basename(filepath)
    return ViewModelLoader(base_dir).load(fname=fname)


def gather_all_references_by_index(h5_file, ref_files):
    refs = h5_file.gather_references()
    for _, gid in refs:
        if not gid:
            continue
        index = load_entity_by_gid(gid)
        h5_file = h5_file_for_index(index)

        if h5_file.path not in ref_files:
            ref_files.append(h5_file.path)

        gather_all_references_by_index(h5_file, ref_files)


def gather_references_of_view_model(gid, base_dir, only_view_models=False):
    """
    Gather in 2 lists all file paths that are referenced by a ViewModel with the given GID stored in base_dir directory.
    If only_view_models=True, returns only ViewModelH5 file paths, otherwise, returns also datatype H5 file paths.
    """

    def load_dts(vm_h5, ref_files):
        uuids = vm_h5.gather_datatypes_references()
        uuid_files = []
        for _, gid in uuids:
            if not gid:
                continue
            index = load_entity_by_gid(gid)
            h5_file = h5_file_for_index(index)
            uuid_files.append(h5_file.path)
            gather_all_references_by_index(h5_file, uuid_files)
        ref_files.extend(uuid_files)

    vm_refs = []
    dt_refs = []
    load_dts_function = None if only_view_models else load_dts
    ViewModelLoader(base_dir).gather_reference_files(gid, vm_refs, dt_refs, load_dts_function)

    if only_view_models:
        return list(set(vm_refs)), None
    else:
        return list(set(vm_refs)), list(set(dt_refs))
