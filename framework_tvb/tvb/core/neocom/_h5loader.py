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
from uuid import UUID
from tvb.basic.neotraits.api import HasTraits
from tvb.core.neotraits.h5 import H5File
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.neocom._registry import Registry

H5_EXTENSION = '.h5'

H5_FILE_NAME_STRUCTURE = '{}_{}.h5'


def get_h5_filename(class_name, gid):
    # type: (str, UUID) -> str
    return H5_FILE_NAME_STRUCTURE.format(class_name, gid.hex)


class Loader(object):
    """
    A default simple loader. Does not do recursive loads. Loads stores just to paths.
    """

    def __init__(self, registry):
        self.registry = registry

    def load(self, source):
        # type: (str) -> HasTraits

        with H5File.from_file(source) as f:
            datatype_cls = self.registry.get_datatype_for_h5file(type(f))
            datatype = datatype_cls()
            f.load_into(datatype)
            return datatype

    def store(self, datatype, destination):
        # type: (HasTraits, str) -> None
        h5file_cls = self.registry.get_h5file_for_datatype(type(datatype))

        with h5file_cls(destination) as f:
            f.store(datatype)


class DirLoader(object):
    """
    A simple recursive loader. Stores all files in a directory.
    You refer to files by their gid
    """

    def __init__(self, base_dir, registry, recursive=False):
        # type: (str, Registry, bool) -> None
        if not os.path.isdir(base_dir):
            raise IOError('not a directory {}'.format(base_dir))

        self.base_dir = base_dir
        self.recursive = recursive
        self.registry = registry

    def _locate(self, gid):
        # type: (uuid.UUID) -> str
        for fname in os.listdir(self.base_dir):
            if fname.endswith(gid.hex + H5_EXTENSION):
                return fname
        raise IOError('could not locate h5 with gid {}'.format(gid))

    def find_file_name(self, gid):
        # type: (typing.Union[uuid.UUID, str]) -> str
        if isinstance(gid, str):
            gid = uuid.UUID(gid)

        fname = self._locate(gid)
        return fname

    def load(self, gid=None, fname=None):
        # type: (typing.Union[uuid.UUID, str], str) -> HasTraits
        """
        Load from file a HasTraits entity. Either gid or fname should be given, or else an error is raised.

        :param gid: optional entity GUID to search for it under self.base_dir
        :param fname: optional file name to search for it under self.base_dir.
        :return: HasTraits instance read from the given location
        """
        if fname is None:
            if gid is None:
                raise ValueError("Neither gid nor filename is provided to load!")
            fname = self.find_file_name(gid)

        sub_dt_refs = []

        with H5File.from_file(os.path.join(self.base_dir, fname)) as f:
            datatype_cls = self.registry.get_datatype_for_h5file(type(f))
            datatype = datatype_cls()
            f.load_into(datatype)

            if self.recursive:
                sub_dt_refs = f.gather_references()

        for traited_attr, sub_gid in sub_dt_refs:
            if sub_gid is not None:
                subdt = self.load(sub_gid)
                setattr(datatype, traited_attr.field_name, subdt)

        return datatype

    def store(self, datatype, fname=None):
        # type: (HasTraits, str) -> None
        h5file_cls = self.registry.get_h5file_for_datatype(type(datatype))
        if fname is None:
            path = self.path_for(h5file_cls, datatype.gid)
        else:
            path = os.path.join(self.base_dir, fname)

        sub_dt_refs = []

        with h5file_cls(path) as f:
            f.store(datatype)
            # Store empty Generic Attributes, so that TVBLoader.load_complete_by_function can be still used
            f.store_generic_attributes(GenericAttributes())

            if self.recursive:
                sub_dt_refs = f.gather_references()

        for traited_attr, sub_gid in sub_dt_refs:
            subdt = getattr(datatype, traited_attr.field_name)
            if subdt is not None:  # Because a non required reference may be not populated
                self.store(subdt)

    def path_for(self, h5_file_class, gid):
        """
        where will this Loader expect to find a file of this format and with this gid
        """
        datatype_cls = self.registry.get_datatype_for_h5file(h5_file_class)
        return self.path_for_has_traits(datatype_cls, gid)

    def _get_has_traits_classname(self, has_traits_class):
        return has_traits_class.__name__

    def path_for_has_traits(self, has_traits_class, gid):

        if isinstance(gid, str):
            gid = uuid.UUID(gid)
        fname = get_h5_filename(self._get_has_traits_classname(has_traits_class), gid)
        return os.path.join(self.base_dir, fname)

    def find_file_for_has_traits_type(self, has_traits_class):

        filename_prefix = self._get_has_traits_classname(has_traits_class)
        for fname in os.listdir(self.base_dir):
            if fname.startswith(filename_prefix) and fname.endswith(H5_EXTENSION):
                return fname
        raise IOError('could not locate h5 for {}'.format(has_traits_class.__name__))


class TVBLoader(object):

    def __init__(self, registry):
        self.file_handler = FilesHelper()
        self.registry = registry

    def path_for_stored_index(self, dt_index_instance):
        # type: (DataType) -> str
        """ Given a Datatype(HasTraitsIndex) instance, build where the corresponding H5 should be or is stored"""
        operation = dao.get_operation_by_id(dt_index_instance.fk_from_operation)
        operation_folder = self.file_handler.get_project_folder(operation.project, str(operation.id))

        gid = uuid.UUID(dt_index_instance.gid)
        h5_file_class = self.registry.get_h5file_for_index(dt_index_instance.__class__)
        fname = get_h5_filename(h5_file_class.file_name_base(), gid)

        return os.path.join(operation_folder, fname)

    def path_for(self, operation_dir, h5_file_class, gid):
        if isinstance(gid, str):
            gid = uuid.UUID(gid)
        fname = get_h5_filename(h5_file_class.file_name_base(), gid)
        return os.path.join(operation_dir, fname)

    def load_from_index(self, dt_index, dt_class=None):
        # type: (DataType, typing.Type[HasTraits]) -> HasTraits
        h5_path = self.path_for_stored_index(dt_index)
        h5_file_class = self.registry.get_h5file_for_index(dt_index.__class__)
        traits_class = dt_class or self.registry.get_datatype_for_index(dt_index.__class__)
        with h5_file_class(h5_path) as f:
            result_dt = traits_class()
            f.load_into(result_dt)
        return result_dt

    def load_complete_by_function(self, file_path, load_ht_function):
        # type: (str, callable) -> (HasTraits, GenericAttributes)
        with H5File.from_file(file_path) as f:
            datatype_cls = self.registry.get_datatype_for_h5file(type(f))
            datatype = datatype_cls()
            f.load_into(datatype)
            ga = f.load_generic_attributes()
            sub_dt_refs = f.gather_references()

        for traited_attr, sub_gid in sub_dt_refs:
            if sub_gid is None:
                continue
            ref_ht = load_ht_function(sub_gid, traited_attr)
            setattr(datatype, traited_attr.field_name, ref_ht)

        return datatype, ga

    def load_with_references(self, file_path):
        def load_ht_function(sub_gid, traited_attr):
            ref_idx = dao.get_datatype_by_gid(sub_gid.hex, load_lazy=False)
            ref_ht = self.load_from_index(ref_idx, traited_attr.field_type)
            return ref_ht

        return self.load_complete_by_function(file_path, load_ht_function)

    def load_with_links(self, file_path):
        def load_ht_function(sub_gid, traited_attr):
            ref_ht = traited_attr.field_type()
            ref_ht.gid = sub_gid
            return ref_ht

        return self.load_complete_by_function(file_path, load_ht_function)
