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
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import os
import uuid
import typing
from tvb.basic.neotraits.api import HasTraits
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neocom._h5loader import Loader, DirLoader, TVBLoader
from tvb.core.neocom._registry import Registry
from tvb.core.neotraits.h5 import H5File

REGISTRY = Registry()


def path_for_stored_index(dt_index_instance):
    # type: (DataType) -> str
    loader = TVBLoader(REGISTRY)
    return loader.path_for_stored_index(dt_index_instance)


def path_for(base_dir, h5_file_class, gid):
    # type: (str, typing.Type[H5File], object) -> str
    loader = TVBLoader(REGISTRY)
    return loader.path_for(base_dir, h5_file_class, gid)


def h5_file_for_index(dt_index_instance):
    # type: (DataType) -> H5File
    h5_path = path_for_stored_index(dt_index_instance)
    h5_class = REGISTRY.get_h5file_for_index(type(dt_index_instance))
    return h5_class(h5_path)


def load_from_index(dt_index, dt_class=None):
    # type: (DataType, typing.Type[HasTraits]) -> HasTraits
    loader = TVBLoader(REGISTRY)
    return loader.load_from_index(dt_index, dt_class)


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


def store_complete(datatype, base_dir):
    # type: (HasTraits, str) -> DataType
    """
    Stores the given HasTraits instance in a h5 file, and fill the Index entity for later storage in DB
    """
    index_class = REGISTRY.get_index_for_datatype(datatype.__class__)
    index_inst = index_class()
    index_inst.fill_from_has_traits(datatype)

    h5_class = REGISTRY.get_h5file_for_datatype(datatype.__class__)
    storage_path = path_for(base_dir, h5_class, datatype.gid)
    with h5_class(storage_path) as f:
        f.store(datatype)
        # Store empty Generic Attributes, in case the file is saved no through ABCAdapter it can still be used
        f.store_generic_attributes(GenericAttributes())

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


def store_to_dir(datatype, base_dir, recursive=False):
    # type: (HasTraits, str, bool) -> None
    """
    Stores the given datatype in the given directory.
    The name and location of the stored file(s) is chosen for you by this function.
    If recursive is true than datatypes referenced by this one are stored as well.
    """
    loader = DirLoader(base_dir, REGISTRY, recursive)
    loader.store(datatype)
