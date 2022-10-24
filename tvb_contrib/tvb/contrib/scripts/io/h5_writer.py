# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from six import string_types
import os
import h5py
import inspect
import numpy

from tvb.core.neocom import h5
from tvb.basic.logger.builder import get_logger
from tvb.basic.readers import H5PY_SUPPORT

from tvb.contrib.scripts.utils.log_error_utils import warning
from tvb.contrib.scripts.utils.data_structures_utils import is_numeric, ensure_list
from tvb.contrib.scripts.utils.file_utils import change_filename_or_overwrite

from .base import Base
from .datatypes_h5 import REGISTRY


h5.REGISTRY = REGISTRY


class H5Writer(Base):

    H5_TYPE_ATTRIBUTE = "Type"
    H5_SUBTYPE_ATTRIBUTE = "Subtype"
    H5_VERSION_ATTRIBUTE = "Version"
    H5_DATE_ATTRIBUTE = "Last_update"

    write_mode = 'w'
    force_overwrite = True

    hfd5_target = None

    def __init__(self, h5_path=""):
        self.logger = get_logger(__name__)
        if H5PY_SUPPORT:
            if len(h5_path):
                self.hfd5_target = h5py.File(h5_path, 'r', libver='latest')
        else:
            self.logger.warning("You need h5py properly installed in order to load from a HDF5 source.")
        self.h5_path = h5_path

    @property
    def _hdf_file(self):
        return self.hfd5_target

    def _set_hdf_file(self, hfile):
        self.hfd5_target = hfile

    @property
    def _fmode(self):
        return self.write_mode

    @property
    def _mode(self):
        return "write"

    @property
    def _mode_past(self):
        return "wrote"

    @property
    def _to_from(self):
        return "to"

    def _open_file(self, type_name=""):
        if self.write_mode == "w":
            self.h5_path = change_filename_or_overwrite(self.h5_path, self.force_overwrite)
        super(H5Writer, self)._open_file(type_name)

    def _determine_datasets_and_attributes(self, object, datasets_size=None):
        datasets_dict = {}
        metadata_dict = {}
        groups_keys = []

        try:
            if isinstance(object, dict):
                dict_object = object
            elif hasattr(object, "to_dict"):
                dict_object = object.to_dict()
            else:
                dict_object = vars(object)
            for key, value in dict_object.items():
                if isinstance(value, numpy.ndarray):
                    # if value.size == 1:
                    #     metadata_dict.update({key: value})
                    # else:
                    datasets_dict.update({key: value})
                    # if datasets_size is not None and value.size == datasets_size:
                    #     datasets_dict.update({key: value})
                    # else:
                    #     if datasets_size is None and value.size > 0:
                    #         datasets_dict.update({key: value})
                    #     else:
                    #         metadata_dict.update({key: value})
                # TODO: check how this works! Be carefull not to include lists and tuples if possible in tvb classes!
                elif isinstance(value, (list, tuple)):
                    warning("Writing %s %s to h5 file as a numpy array dataset !" % (value.__class__, key), self.logger)
                    datasets_dict.update({key: numpy.array(value)})
                else:
                    if is_numeric(value) or isinstance(value, str):
                        metadata_dict.update({key: value})
                    elif callable(value):
                        metadata_dict.update({key: inspect.getsource(value)})
                    elif value is None:
                        continue
                    else:
                        groups_keys.append(key)
        except Exception as e:
            msg = "Failed to decompose group object: " + str(object) + "!" + "\nThe error was\n%s" % str(e)
            try:
                self.logger.info(str(object.__dict__))
            except:
                msg += "\n It has no __dict__ attribute!"
            self.logger.warning(msg)

        return datasets_dict, metadata_dict, groups_keys

    def _write_dicts_at_location(self, datasets_dict, metadata_dict, location):
        for key, value in datasets_dict.items():
            try:
                try:
                    location.create_dataset(key, data=value)
                except:
                    location.create_dataset(key, data=numpy.str(value))
            except Exception as e:
                self.logger.warning("Failed to write to %s dataset %s %s:\n%s !\nThe error was:\n%s" %
                                    (str(location), value.__class__, key, str(value), str(e)))

        for key, value in metadata_dict.items():
            try:
                location.attrs.create(key, value)
            except Exception as e:
                self.logger.warning("Failed to write to %s attribute %s %s:\n%s !\nThe error was:\n%s" %
                                    (str(location), value.__class__, key, str(value), str(e)))
        return location

    def _prepare_object_for_group(self, group, object, h5_type_attribute="", nr_regions=None,
                                  regress_subgroups=True):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, h5_type_attribute)
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)
        datasets_dict, metadata_dict, subgroups = self._determine_datasets_and_attributes(object, nr_regions)
        # If empty return None
        if len(datasets_dict) == len(metadata_dict) == len(subgroups) == 0:
            if isinstance(group, h5py._hl.files.File):
                if regress_subgroups:
                    return group
                else:
                    return group, subgroups
            else:
                return None
        else:
            if len(datasets_dict) > 0 or len(metadata_dict) > 0:
                if isinstance(group, h5py._hl.files.File):
                    group = self._write_dicts_at_location(datasets_dict, metadata_dict, group)
                else:
                    self._write_dicts_at_location(datasets_dict, metadata_dict, group)
            # Continue recursively going deeper in the object
            if regress_subgroups:
                for subgroup in subgroups:
                    if isinstance(object, dict):
                        child_object = object.get(subgroup, None)
                    else:
                        child_object = getattr(object, subgroup, None)
                    if child_object is not None:
                        group.require_group(subgroup)
                        temp = self._prepare_object_for_group(group[subgroup], child_object,
                                                              h5_type_attribute, nr_regions)
                        # If empty delete it
                        if temp is None or (len(temp.keys()) == 0 and len(temp.attrs.keys()) == 0):
                            del group[subgroup]

                return group
            else:
                return group, subgroups

    def write_object(self, object, h5_type_attribute="", nr_regions=None,
                     path=None, close_file=True):
        """
                :param object: object to write recursively in H5
                :param path: H5 path to be written
        """
        self._assert_file(path, object.__class__.__name__)
        try:
            self.hfd5_target = self._prepare_object_for_group(self.hfd5_target, object, h5_type_attribute, nr_regions)
            self._log_success_or_warn(None, object.__class__.__name__)
        except Exception as e:
            self._log_success_or_warn(e, object.__class__.__name__)
        self._close_file(close_file)

    def write_list_of_objects(self, list_of_objects, path=None, close_file=True):
        self._assert_file(path, "List of objects")
        try:
            for idict, object in enumerate(list_of_objects):
                idict_str = str(idict)
                h5_file.create_group(idict_str)
                self.write_object(object, h5_file=h5_file[idict_str], close_file=False)
            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("List of objects"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_("list"))
            self._log_success_or_warn(None, "List of objects")
        except Exceptions as e:
            self._log_success_or_warn(e, "List of objects")
        self._close_file(close_file)

    def _convert_sequences_of_strings(self, sequence):
        new_sequence = []
        for val in ensure_list(sequence):
            if isinstance(val, string_types):
                new_sequence.append(numpy.string_(val))
            elif isinstance(val, (numpy.ndarray, tuple, list)):
                new_sequence.append(self._convert_sequences_of_strings(val))
            else:
                new_sequence.append(val)
        return numpy.array(new_sequence)

    def _write_dictionary_to_group(self, dictionary, group):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, "Dictionary")
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, dictionary.__class__.__name__)
        for key, value in dictionary.items():
            try:
                if isinstance(value, (numpy.ndarray, list, tuple)) and len(value) > 0:
                    new_value = numpy.array(value)
                    if not numpy.issubdtype(value.dtype, numpy.number):
                        new_value = self._convert_sequences_of_strings(new_value)
                    group.create_dataset(key, data=new_value)
                else:
                    if callable(value):
                        group.attrs.create(key, inspect.getsource(value))
                    elif isinstance(value, dict):
                        group.create_group(key)
                        self._write_dictionary_to_group(value, group[key])
                    elif value is None:
                        continue
                    else:
                        group.attrs.create(key, numpy.string_(value))
            except Exception as e:
                self.logger.warning("Did not manage to write %s to h5 file %s !\nThe error was:\n%s"
                                    % (key, str(group), str(e)))

    def write_dictionary(self, dictionary, path=None, close_file=True):
        """
        :param dictionary: dictionary/ies to write recursively in H5
        :param path: H5 path to be written
        Use this function only if you have to write dictionaries of data (scalars and arrays or lists of scalars,
        or of more such dictionaries recursively
        """
        self._assert_file(path, "Dictionary")
        try:
            self._write_dictionary_to_group(dictionary, h5_file)
            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Dictionary"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(dictionary.__class__.__name__))
            self._log_success_or_warn(None, "Dictionary")
        except Exceptions as e:
            self._log_success_or_warn(e, "Dictionary")
        self._close_file(close_file)

    def write_list_of_dictionaries(self, list_of_dicts, path=None, close_file=True):
        self._assert_file(path, "List of Dictionaries")
        try:
            for idict, dictionary in enumerate(list_of_dicts):
                idict_str = str(idict)
                h5_file.create_group(idict_str)
                self._write_dictionary_to_group(dictionary, h5_file[idict_str])
            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("List of dictionaries"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_("list"))
            self._log_success_or_warn(None, "List of Dictionaries")
        except Exceptions as e:
            self._log_success_or_warn(e, "List of Dictionaries")
        self._close_file(close_file)

    def write_tvb_to_h5(self, datatype, path=None, recursive=True, force_overwrite=True):
        if path is None:
            path = self.h5_path
        else:
            self.h5_path = path
        if path.endswith("h5"):
            # It is a file path:
            dirpath = os.path.dirname(self.h5_path)
            if os.path.isdir(dirpath):
                self.h5_path = change_filename_or_overwrite(self.h5_path, force_overwrite)
            else:
                os.mkdir(dirpath)
        else:
            if not os.path.isdir(self.h5_path):
                os.mkdir(self.h5_path)
            self.h5_path = os.path.join(self.h5_path, datatype.title + ".h5")
            self.h5_path = change_filename_or_overwrite(self.h5_path, self.force_overwrite)
        try:
            h5.store(datatype, self.h5_path, recursive)
            self._log_success_or_warn(None, datatype.title)
        except Exceptions as e:
            self._log_success_or_warn(e, datatype.title)
        return self.h5_path
