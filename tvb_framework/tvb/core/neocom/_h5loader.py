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
from datetime import datetime

from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage import dao
from tvb.core.neocom._registry import Registry
from tvb.core.neotraits.h5 import H5File, ViewModelH5
from tvb.core.neotraits.view_model import ViewModel
from tvb.core.utils import string2date, date2string
from tvb.storage.storage_interface import StorageInterface


class Loader(object):
    """
    A default simple loader. Does not do recursive loads. Loads stores just to paths.
    """

    def __init__(self, registry):
        self.registry = registry

    def load(self, source):
        # type: (str) -> HasTraits

        with H5File.from_file(source) as f:
            datatype_cls = self.registry.get_datatype_for_h5file(f)
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
        self.log = get_logger(__name__)

        if not os.path.isdir(base_dir):
            raise IOError('not a directory {}'.format(base_dir))

        self.base_dir = base_dir
        self.recursive = recursive
        self.registry = registry

    def _locate(self, gid):
        # type: (uuid.UUID) -> str
        for fname in os.listdir(self.base_dir):
            if fname.endswith(gid.hex + StorageInterface.TVB_STORAGE_FILE_EXTENSION):
                fpath = os.path.join(self.base_dir, fname)
                return fpath
        raise IOError('could not locate h5 with gid {}'.format(gid))

    def find_file_by_gid(self, gid):
        # type: (typing.Union[uuid.UUID, str]) -> str
        if isinstance(gid, str):
            gid = uuid.UUID(gid)
        fpath = self._locate(gid)
        self.log.debug("Computed path for H5 is: {}".format(fpath))
        return fpath

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
            fname = self.find_file_by_gid(gid)
        elif not os.path.isabs(fname):
            fname = os.path.join(self.base_dir, fname)

        sub_dt_refs = []

        with H5File.from_file(fname) as f:
            datatype_cls = self.registry.get_datatype_for_h5file(f)
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
        # type: (HasTraits, str) -> str
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

        return path

    def path_for(self, h5_file_class, gid):
        """
        where will this Loader expect to find a file of this format and with this gid
        """
        datatype_cls = self.registry.get_base_datatype_for_h5file(h5_file_class)
        return self.path_for_has_traits(datatype_cls, gid)

    def _get_has_traits_classname(self, has_traits_class):
        return has_traits_class.__name__

    def path_for_has_traits(self, has_traits_class, gid):

        if isinstance(gid, str):
            gid = uuid.UUID(gid)
        fname = StorageInterface().get_filename(self._get_has_traits_classname(has_traits_class), gid)
        return os.path.join(self.base_dir, fname)

    def find_file_for_has_traits_type(self, has_traits_class):

        filename_prefix = self._get_has_traits_classname(has_traits_class)
        for fname in os.listdir(self.base_dir):
            if fname.startswith(filename_prefix) and fname.endswith(StorageInterface.TVB_STORAGE_FILE_EXTENSION):
                return fname
        raise IOError('could not locate h5 for {}'.format(has_traits_class.__name__))


class TVBLoader(object):
    """
    A loader for HasTraits objects.
    Works with the TVB database and the TVB storage folder structure to identify and load datatypes starting from their
    corresponding HasTraitsIndex or a file path.
    Intended for usage in tvb-framework.
    """

    def __init__(self, registry):
        self.storage_interface = StorageInterface()
        self.registry = registry

    def path_for_stored_index(self, dt_index_instance):
        # type: (DataType) -> str
        """ Given a Datatype(HasTraitsIndex) instance, build where the corresponding H5 should be or is stored"""
        if hasattr(dt_index_instance, 'fk_simulation'):
            # In case of BurstConfiguration the operation id is on fk_simulation
            op_id = dt_index_instance.fk_simulation
        else:
            op_id = dt_index_instance.fk_from_operation
        operation = dao.get_operation_by_id(op_id)
        operation_folder = self.storage_interface.get_project_folder(operation.project.name, str(operation.id))

        gid = uuid.UUID(dt_index_instance.gid)
        h5_file_class = self.registry.get_h5file_for_index(dt_index_instance.__class__)
        fname = self.storage_interface.get_filename(h5_file_class.file_name_base(), gid)

        return os.path.join(operation_folder, fname)

    def path_for(self, op_id, h5_file_class, gid, project_name, dt_class):
        return self.storage_interface.path_for(op_id, h5_file_class, gid, project_name, dt_class)

    def path_by_dir(self, base_dir, h5_file_class, gid, dt_class):
        return self.storage_interface.path_by_dir(base_dir, h5_file_class, gid, dt_class)

    def load_from_index(self, dt_index):
        # type: (DataType) -> HasTraits
        h5_path = self.path_for_stored_index(dt_index)
        h5_file_class = self.registry.get_h5file_for_index(dt_index.__class__)
        traits_class = self.registry.get_datatype_for_index(dt_index)
        with h5_file_class(h5_path) as f:
            result_dt = traits_class()
            f.load_into(result_dt)
        return result_dt

    def load_complete_by_function(self, file_path, load_ht_function):
        # type: (str, callable) -> (HasTraits, GenericAttributes)
        with H5File.from_file(file_path) as f:
            try:
                datatype_cls = self.registry.get_datatype_for_h5file(f)
            except KeyError:
                datatype_cls = f.determine_datatype_from_file()
            datatype = datatype_cls()
            f.load_into(datatype)
            ga = f.load_generic_attributes()
            sub_dt_refs = f.gather_references(datatype_cls)

        for traited_attr, sub_gid in sub_dt_refs:
            if sub_gid is None:
                continue
            is_monitor = False
            if isinstance(sub_gid, list):
                sub_gid = sub_gid[0]
                is_monitor = True
            ref_ht = load_ht_function(sub_gid, traited_attr)
            if is_monitor:
                ref_ht = [ref_ht]
            setattr(datatype, traited_attr.field_name, ref_ht)

        return datatype, ga

    def load_with_references(self, file_path):
        def load_ht_function(sub_gid, traited_attr):
            ref_idx = dao.get_datatype_by_gid(sub_gid.hex, load_lazy=False)
            ref_ht = self.load_from_index(ref_idx)
            return ref_ht

        return self.load_complete_by_function(file_path, load_ht_function)

    def load_with_links(self, file_path):
        def load_ht_function(sub_gid, traited_attr):
            # Used traited_attr.default for cases similar to ProjectionMonitor which has obsnoise of type Noise and
            # it cannot be instantiated due to abstract methods, while the default is Additive()
            ref_ht = traited_attr.default or traited_attr.field_type()
            ref_ht.gid = sub_gid
            return ref_ht

        return self.load_complete_by_function(file_path, load_ht_function)


class ViewModelLoader(DirLoader):
    """
    A recursive loader for ViewModel objects.
    Stores all files in one directory specified at initialization time.
    Does not access the TVB database. Does not take into consideration the TVB storage folder structure.
    Stores every linked HasTraits in a file, but not datatypes that are already stored in TVB storage.
    Intended for usage within tvb-framework to store view models in H5 files.
    """

    def __init__(self, base_dir, registry=None):
        super().__init__(base_dir, registry)

    def get_class_path(self, vm):
        return vm.__class__.__module__ + '.' + vm.__class__.__name__

    def store(self, view_model, fname=None):
        # type: (ViewModel, str) -> str
        """
        Completely store any ViewModel object to the directory specified by self.base_dir.
        Works recursively for view models that are serialized in multiple files (eg. SimulatorAdapterModel)
        """
        if fname is None:
            h5_path = self.path_for_has_traits(type(view_model), view_model.gid)
        else:
            h5_path = os.path.join(self.base_dir, fname)
        with ViewModelH5(h5_path, view_model) as h5_file:
            self._store(h5_file, view_model)
        return h5_path

    def _store(self, file, view_model):
        file.store(view_model)
        file.type.store(self.get_class_path(view_model))
        file.create_date.store(date2string(datetime.now()))
        if hasattr(view_model, "generic_attributes"):
            file.store_generic_attributes(view_model.generic_attributes)
        else:
            # For HasTraits not inheriting from ViewModel (e.g. Linear)
            file.store_generic_attributes(GenericAttributes())

        references = file.gather_references()
        for trait_attr, gid in references:
            if not gid:
                continue
            model_attr = getattr(view_model, trait_attr.field_name)
            if isinstance(gid, list):
                for idx, sub_gid in enumerate(gid):
                    self.store(model_attr[idx])
            else:
                self.store(model_attr)

    def load(self, gid=None, fname=None):
        # type: (typing.Union[uuid.UUID, str], str) -> ViewModel
        """
        Load a ViewModel object by reading the H5 file with the given GID, from the directory self.base_dir
        """
        if fname is None:
            if gid is None:
                raise ValueError("Neither gid nor filename is provided to load!")
            fname = self.find_file_by_gid(gid)
        else:
            fname = os.path.join(self.base_dir, fname)

        view_model_class = H5File.determine_type(fname)
        view_model = view_model_class()

        with ViewModelH5(fname, view_model) as h5_file:
            self._load(h5_file, view_model)
        return view_model

    def _load(self, file, view_model):
        file.load_into(view_model)
        references = file.gather_references()
        view_model.create_date = string2date(file.create_date.load())
        view_model.generic_attributes = file.load_generic_attributes()
        for trait_attr, gid in references:
            if not gid:
                continue
            if isinstance(gid, list):
                loaded_ref = []
                for idx, sub_gid in enumerate(gid):
                    ref = self.load(sub_gid)
                    loaded_ref.append(ref)
            else:
                loaded_ref = self.load(gid)
            setattr(view_model, trait_attr.field_name, loaded_ref)

    def gather_reference_files(self, gid, vm_ref_files, dt_ref_files, load_dts=None):
        vm_path = self.find_file_by_gid(gid)
        vm_ref_files.append(vm_path)
        view_model_class = H5File.determine_type(vm_path)
        view_model = view_model_class()

        with ViewModelH5(vm_path, view_model) as vm_h5:
            references = vm_h5.gather_references()

            for _, gid in references:
                if not gid:
                    continue
                if isinstance(gid, (list, tuple)):
                    for list_gid in gid:
                        self.gather_reference_files(list_gid, vm_ref_files, dt_ref_files, load_dts)
                else:
                    self.gather_reference_files(gid, vm_ref_files, dt_ref_files, load_dts)
            if load_dts:
                load_dts(vm_h5, dt_ref_files)


class DtLoader(ViewModelLoader):
    """
    A recursive loader for datatypes (HasTraits).
    Stores all files in one directory specified at initialization time.
    Does not access the TVB database. Does not take into consideration the TVB storage folder structure.
    Stores every linked HasTraits object in a file (even datatypes which might already exist in the TVB storage).
    Intended for storing tvb-library results in H5 files.
    """

    def __init__(self, base_dir, registry):
        super().__init__(base_dir, registry)

    def load(self, gid=None, fname=None):
        # type: (typing.Union[uuid.UUID, str], str) -> ViewModel
        """
        Load a HasTraits datatype object by reading the H5 file with the given GID, from the directory self.base_dir
        """
        if fname is None:
            if gid is None:
                raise ValueError("Neither gid nor filename is provided to load!")
            fname = self.find_file_by_gid(gid)
        else:
            fname = os.path.join(self.base_dir, fname)

        ht_datatype_class = H5File.determine_type(fname)
        ht_datatype = ht_datatype_class()

        ht_datatype_h5 = self.registry.get_h5file_for_datatype(ht_datatype.__class__)
        if ht_datatype_h5 != H5File:
            with ht_datatype_h5(fname) as file:
                self._load(file, ht_datatype)
        else:
            with ViewModelH5(fname, ht_datatype) as h5_file:
                self._load(h5_file, ht_datatype)
        return ht_datatype

    def store(self, ht_datatype, fname=None):
        # type: (HasTraits, str) -> str
        """
        Completely store any HasTraits datatype object to the directory specified by self.base_dir.
        Works recursively for datatypes that are serialized in multiple files (eg. Simulator)
        """
        if fname is None:
            h5_path = self.path_for_has_traits(type(ht_datatype), ht_datatype.gid)
        else:
            h5_path = os.path.join(self.base_dir, fname)
        ht_datatype_h5 = self.registry.get_h5file_for_datatype(ht_datatype.__class__)
        if ht_datatype_h5 != H5File:
            with ht_datatype_h5(h5_path) as file:
                self._store(file, ht_datatype)
        else:
            with ViewModelH5(h5_path, ht_datatype) as h5_file:
                self._store(h5_file, ht_datatype)
        return h5_path
