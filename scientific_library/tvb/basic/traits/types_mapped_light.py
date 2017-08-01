# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Mapped super-classes are defined here.

Important:

- Type - traited, possible mapped to db *col*
- MappedType - traited, mapped to db *table*


.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <mw@eml.cc>

"""

import six
import numpy
from scipy import sparse
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.util import get
from tvb.basic.traits.core import Type
from tvb.basic.traits.types_basic import DType


class MappedTypeLight(Type):
    """
    Light base class for all entities which are about to be mapped in storage.
    Current light implementation is to be used with the scientific-library stand-alone mode.
    """

    METADATA_EXCLUDE_PARAMS = ['id', 'LINKS', 'fk_datatype_group', 'disk_size',
                               'fk_from_operation', 'parent_operation', 'fk_parent_burst']

    ### Constants when retrieving meta-data about Array attributes on the current instance.
    METADATA_ARRAY_MAX = "Maximum"
    METADATA_ARRAY_MIN = "Minimum"
    METADATA_ARRAY_MEAN = "Mean"
    METADATA_ARRAY_VAR = "Variance"
    METADATA_ARRAY_MIN_NON_ZERO = "Min. non zero"
    METADATA_ARRAY_MAX_NON_ZERO = "Max. non zero"
    METADATA_ARRAY_MEAN_NON_ZERO = "Mean non zero"
    METADATA_ARRAY_VAR_NON_ZERO = "Var. non zero"
    METADATA_ARRAY_SHAPE = "Shape"
    _METADATA_ARRAY_SIZE = "Size"
    _METADATA_ARRAY_SIZE_NON_ZERO = "Size"

    DEFAULT_STORED_ARRAY_METADATA = [METADATA_ARRAY_MAX, METADATA_ARRAY_MIN, METADATA_ARRAY_MEAN,
                                     METADATA_ARRAY_VAR, METADATA_ARRAY_SHAPE]
    DEFAULT_WITH_ZERO_METADATA = [METADATA_ARRAY_MAX, METADATA_ARRAY_MIN, METADATA_ARRAY_MEAN,
                                  METADATA_ARRAY_VAR, METADATA_ARRAY_SHAPE, METADATA_ARRAY_MIN_NON_ZERO,
                                  METADATA_ARRAY_MEAN_NON_ZERO, METADATA_ARRAY_VAR_NON_ZERO]

    METADATA_FORMULAS = {METADATA_ARRAY_MAX: '$ARRAY$.max()',
                         METADATA_ARRAY_MIN: '$ARRAY$.min()',
                         METADATA_ARRAY_MEAN: '$ARRAY$.mean()',
                         METADATA_ARRAY_VAR: '$ARRAY$.var()',
                         METADATA_ARRAY_MAX_NON_ZERO: '$ARRAY$[$MASK$.nonzero()].max()',
                         METADATA_ARRAY_MIN_NON_ZERO: '$ARRAY$[$MASK$.nonzero()].min()',
                         METADATA_ARRAY_MEAN_NON_ZERO: '$ARRAY$[$MASK$.nonzero()].mean()',
                         METADATA_ARRAY_VAR_NON_ZERO: '$ARRAY$[$MASK$.nonzero()].var()',
                         METADATA_ARRAY_SHAPE: '$ARRAY$.shape()'}

    logger = get_logger(__name__)


    def __init__(self, **kwargs):
        super(MappedTypeLight, self).__init__(**kwargs)
        self._current_metadata = dict()


    @classmethod
    def from_file(cls, source_file="", instance=None):
        raise NotImplementedError("This DataType can not be used with load_default=True")


    def accepted_filters(self):
        """
        Just offer dummy functionality in library mode.
        """
        return {}

    @property
    def display_name(self):
        """
        To be implemented in each sub-class which is about to be displayed in UI,
        and return the text to appear.
        """
        return self.__class__.__name__


    def get_info_about_array(self, array_name, included_info=None, mask_array_name=None, key_suffix=''):
        """
        :return: dictionary {label: value} about an attribute of type mapped.Array
                 Generic information, like Max/Min/Mean/Var are to be retrieved for this array_attr
        """
        included_info = included_info or {}
        summary = self._get_summary_info(array_name, included_info, mask_array_name, key_suffix)
        ### Before return, prepare names for UI display.                
        result = dict()
        for key, value in six.iteritems(summary):
            result[array_name.capitalize().replace("_", " ") + " - " + key] = value
        return result


    def _get_summary_info(self, array_name, included_info, mask_array_name, key_suffix):
        """
        Get a summary from the metadata of the current array.
        :return: dictionary {label: value} about an attribute of type mapped.Array, with information like max/mean/etc
        """
        summary = dict()

        array_attr = getattr(self, array_name)
        if mask_array_name is not None:
            mask_attr = getattr(self, mask_array_name)
        else:
            mask_attr = array_attr

        if isinstance(array_attr, numpy.ndarray) and isinstance(mask_attr, numpy.ndarray):
            for key in included_info:
                if key in self.METADATA_FORMULAS:
                    summary[key + key_suffix] = eval(self.METADATA_FORMULAS[key].replace(
                        "$ARRAY$", "array_attr").replace("$MASK$", "mask_attr"))
                else:
                    self.logger.warning("Not supported meta-data will be ignored " + str(key))
        return summary


    def get_data_shape(self, data_name):
        """
        This method reads data-shape from the given data set
            ::param data_name: Name of the attribute from where to read size
            ::return: a shape tuple
        """
        array_data = getattr(self, data_name)
        if hasattr(array_data, 'shape'):
            return getattr(array_data, 'shape')
        self.logger.warning("Could not find 'shape' attribute on " + str(data_name) + " returning empty shape!!")
        return ()


    def get_data(self, data_name, data_slice=None):
        """
        Get sliced portion of an attribute of type numpy array.
        """
        array_data = getattr(self, data_name)
        return array_data[data_slice]


class Array(Type):
    """
    Traits type that wraps a NumPy NDArray.

    Initialization requires at least shape, and when not given, will be set to (), an empty, 0-dimension array.
    """

    wraps = numpy.ndarray
    dtype = DType()
    defaults = ((0,), {})
    data = None
    stored_metadata = [key for key in MappedTypeLight.DEFAULT_STORED_ARRAY_METADATA]
    logger = get_logger(__name__)


    @property
    def shape(self):
        """  
        Property SHAPE for the wrapped array.
        """
        return self.data.shape


    @property
    def array_path(self):
        """  
        Property PATH relative.
        """
        return self.trait.name


    def __get__(self, inst, cls):
        """
        When an attribute of class Array is retrieved on another class.
        :param inst: It is a MappedType instance
        :param cls: MappedType subclass. When 'inst' is None and only 'cls' is passed, we do not read from storage,
                    but return traited attribute.
        :return: value of type self.wraps
        :raise Exception: when read could not be executed, Or when used GET with incompatible attributes (e.g. chunks).
        """
        if inst is None:
            return self

        if self.trait.bound:
            return self._get_cached_data(inst)
        else:
            return self


    def __set__(self, inst, value):
        """
        This is called when an attribute of type Array is set on another class instance.
        :param inst: It is a MappedType instance
        :param value: expected to be of type self.wraps
        :raise Exception: When incompatible type of value is set
        """
        self._put_value_on_instance(inst, self.array_path)
        if isinstance(value, list):
            value = numpy.array(value)
        elif type(value) in (int, float):
            value = numpy.array([value])

        setattr(inst, '__' + self.trait.name, value)


    def _get_cached_data(self, inst):
        """
        Just read from instance since we don't have storage in library mode.
        """
        return get(inst, '__' + self.trait.name, None)


    def log_debug(self, owner=""):
        """
        Simple access to debugging info on a traited array, usage ::
            obj.trait["array_name"].log_debug(owner="obj")
            
        or ::
            self.trait["array_name"].log_debug(owner=self.__class__.__name__)
        """
        name = ".".join((owner, self.trait.name))
        sts = str(self.__class__)
        if self.trait.value is not None and self.trait.value.size != 0:
            shape = str(self.trait.value.shape)
            dtype = str(self.trait.value.dtype)
            tvb_dtype = str(self.trait.value.dtype)
            has_nan = str(numpy.isnan(self.trait.value).any())
            array_max = str(self.trait.value.max())
            array_min = str(self.trait.value.min())
            self.logger.debug("%s: %s shape: %s" % (sts, name, shape))
            self.logger.debug("%s: %s actual dtype: %s" % (sts, name, dtype))
            self.logger.debug("%s: %s tvb dtype: %s" % (sts, name, tvb_dtype))
            self.logger.debug("%s: %s has NaN: %s" % (sts, name, has_nan))
            self.logger.debug("%s: %s maximum: %s" % (sts, name, array_max))
            self.logger.debug("%s: %s minimum: %s" % (sts, name, array_min))
        else:
            self.logger.debug("%s: %s is Empty" % (sts, name))


class SparseMatrix(Array):
    """
    Map a big matrix.
    Will require storage in File Structure.
    """
    wraps = sparse.csc_matrix
    defaults = (((1, 1),), {'dtype': numpy.float64})
    logger = get_logger(__name__)


    def log_debug(self, owner=""):
        """
        Simple access to debugging info on a traited sparse matrix, usage ::
            obj.trait["sparse_matrix_name"].log_debug(owner="obj")
            
        or ::
            self.trait["sparse_matrix_name"].log_debug(owner=self.__class__.__name__)
        """
        name = ".".join((owner, self.trait.name))
        sts = str(self.__class__)
        if self.trait.value.size != 0:
            shape = str(self.trait.value.shape)
            sparse_format = str(self.trait.value.format)
            nnz = str(self.trait.value.nnz)
            dtype = str(self.trait.value.dtype)
            array_max = str(self.trait.value.data.max())
            array_min = str(self.trait.value.data.min())
            self.logger.debug("%s: %s shape: %s" % (sts, name, shape))
            self.logger.debug("%s: %s format: %s" % (sts, name, sparse_format))
            self.logger.debug("%s: %s number of non-zeros: %s" % (sts, name, nnz))
            self.logger.debug("%s: %s dtype: %s" % (sts, name, dtype))
            self.logger.debug("%s: %s maximum: %s" % (sts, name, array_max))
            self.logger.debug("%s: %s minimum: %s" % (sts, name, array_min))
        else:
            self.logger.debug("%s: %s is Empty" % (sts, name))
