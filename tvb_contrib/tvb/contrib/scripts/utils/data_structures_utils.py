# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import re
import itertools
import numpy as np
from copy import deepcopy
from collections.abc import Hashable
from six import string_types
from collections import OrderedDict
from tvb.contrib.scripts.utils.log_error_utils import warning, raise_value_error, raise_import_error
from tvb.basic.logger.builder import get_logger

logger = get_logger(__name__)


class CalculusConfig(object):
    # Normalization configuration
    WEIGHTS_NORM_PERCENT = 99

    # If True a plot will be generated to choose the number of eigenvalues to keep
    INTERACTIVE_ELBOW_POINT = False

    MIN_SINGLE_VALUE = np.finfo("single").min
    MAX_SINGLE_VALUE = np.finfo("single").max
    MAX_INT_VALUE = np.iinfo(np.int64).max
    MIN_INT_VALUE = np.iinfo(np.int64).max


def is_numeric(value):
    return isinstance(value, (float, np.float_, np.float64, np.float32, np.float16, np.float128,
                              int, np.int_, np.int0, np.int8, np.int16, np.int32, np.int64,
                              complex, np.complex, np.complex64, np.complex128, np.complex256,
                              np.long, np.number))


def is_integer(value):
    return isinstance(value, (int, np.int_, np.intp, np.int8, np.int16, np.int32, np.int64))


def is_float(value):
    return isinstance(value, (float, np.float_, np.float64, np.float32, np.float16, np.float128))


def vector2scalar(x):
    if not (isinstance(x, np.ndarray)):
        return x
    else:
        y = np.squeeze(x)
    if all(y.squeeze() == y[0]):
        return y[0]
    else:
        return reg_dict(x)


def list_of_strings_to_string(lstr, sep=","):
    result_str = lstr[0]
    for s in lstr[1:]:
        result_str += sep + s
    return result_str


def dict_str(d):
    s = "{"
    for key, value in d.items():
        s += ("\n" + key + ": " + str(value))
    s += "}"
    return s


def isequal_string(a, b, case_sensitive=False):
    if case_sensitive:
        return a == b
    else:
        try:
            return a.lower() == b.lower()
        except AttributeError:
            logger.warning("Case sensitive comparison!")
            return a == b


def split_string_text_numbers(ls):
    items = []
    for s in ensure_list(ls):
        match = re.findall('(\d+|\D+)', s)
        if match:
            items.append(tuple(match[:2]))
    return items


def join_labels_indices_dict(d):
    out_list = []
    for label, ind_list in sort_dict(d).items():
        for ind in ind_list:
            out_list.append(label + "%d" % ind)
    return out_list


def construct_import_path(path, package=None):
    path = path.split(".py")[0]
    if isinstance(package, string_types):
        start = path.find(package)
    else:
        start = 0
    return path[start:].replace("/", ".")


def format_all_numbers_in_strings(input_strings, num_integers=7, num_decimals=6):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    integer_format = "%0" + str(num_integers) + "d"
    num_total = num_integers + num_decimals + 1
    float_format = "%0" + str(num_total) + "." + str(num_decimals) + "f"
    output_strings = []
    for inp_str in ensure_list(input_strings):
        number_strings = rx.findall(inp_str)
        string_parts = rx.split(inp_str)
        out_str = ""
        for string_part, num_str in zip(string_parts, number_strings):
            try:
                out_str += string_part + integer_format % int(num_str)
            except:
                out_str += string_part + float_format % float(num_str)
        if len(string_parts) > len(number_strings):
            out_str += string_parts[-1]
        output_strings.append(out_str)
    if len(output_strings) == 1:
        return output_strings[0]
    return output_strings


def formal_repr(instance, attr_dict, sort_dict_flag=False):
    """ A formal string representation for an object.
    :param attr_dict: dictionary attribute_name: attribute_value
    :param instance:  Instance to read class name from it
    """
    class_name = instance.__class__.__name__
    formal = class_name + "{"
    if sort_dict_flag:
        attr_dict = sort_dict(attr_dict)
    for key, val in attr_dict.items():
        if isinstance(val, dict):
            formal += "\n" + key + "=["
            for key2, val2 in val.items():
                formal += "\n" + str(key2) + " = " + str(val2)
            formal += "]"
        else:
            formal += "\n" + str(key) + " = " + str(val)
    return formal + "}"


def obj_to_dict(obj):
    """
    :param obj: Python object to introspect
    :return: dictionary after recursively taking obj fields and their values
    """
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, (np.float32,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, list):
        ret = []
        for val in obj:
            ret.append(obj_to_dict(val))
        return ret
    ret = {}
    for key in obj.__dict__:
        val = getattr(obj, key, None)
        ret[key] = obj_to_dict(val)
    return ret


def reg_dict(x, lbl=None, sort=None):
    """
    :x: a list or np vector
    :lbl: a list or np vector of labels
    :return: dictionary
    """
    if not (isinstance(x, (str, int, float, list, np.ndarray))):
        return x
    else:
        if not (isinstance(x, list)):
            x = np.squeeze(x)
        x_no = len(x)
        if not (isinstance(lbl, (list, np.ndarray))):
            lbl = np.repeat('', x_no)
        else:
            lbl = np.squeeze(lbl)
        labels_no = len(lbl)
        total_no = min(labels_no, x_no)
        if x_no <= labels_no:
            if sort == 'ascend':
                ind = np.argsort(x).tolist()
            elif sort == 'descend':
                ind = np.argsort(x)
                ind = ind[::-1].tolist()
            else:
                ind = range(x_no)
        else:
            ind = range(total_no)
        d = OrderedDict()
        for i in ind:
            d[str(i) + '.' + str(lbl[i])] = x[i]
        if labels_no > total_no:
            ind_lbl = np.delete(np.array(range(labels_no)), ind).tolist()
            for i in ind_lbl:
                d[str(i) + '.' + str(lbl[i])] = None
        if x_no > total_no:
            ind_x = np.delete(np.array(range(x_no)), ind).tolist()
            for i in ind_x:
                d[str(i) + '.'] = x[i]
        return d


def sort_dict(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def switch_levels_of_dicts_of_dicts(d):
    keys = d.values()[0].keys()
    return {key: {k: d[k][key] for k in d if key in d[k]} for key in keys}


def dicts_of_lists(dictionary, n=1):
    for key, value in dictionary.items():
        dictionary[key] = ensure_list(dictionary[key])
        if len(dictionary[key]) == 1 and n > 1:
            dictionary[key] = dictionary[key] * n
    return dictionary


def iterable_to_dict(obj):
    d = OrderedDict()
    for ind, value in enumerate(obj):
        d["%02d" % ind] = value
    return d


def dict_to_list_or_tuple(dictionary, output_obj="list"):
    dictionary = sort_dict(dictionary)
    output = dictionary.values()
    if output_obj == "tuple":
        output = tuple(output)
    return output


def list_of_dicts_to_dict_of_tuples(lst):
    return dict(zip(lst[0], zip(*list([d.values() for d in lst]))))


def list_of_dicts_to_dict_of_lists(lst):
    d = list_of_dicts_to_dict_of_tuples(lst)
    for key, val in d.items():
        d[key] = list(val)
    return d


def list_of_dicts_to_dicts_of_ndarrays(lst, shape=None):
    d = list_of_dicts_to_dict_of_tuples(ensure_list(lst))
    if isinstance(shape, tuple):
        for key, val in d.items():
            d[key] = np.reshape(np.stack(d[key]), shape)
    else:
        for key, val in d.items():
            d[key] = np.stack(d[key])
            for sh in d[key].shape[:0:-1]:
                if sh == 1:
                    d[key] = np.squeeze(d[key], axis=-1)
                else:
                    break
    return d


def arrays_of_dicts_to_dicts_of_ndarrays(arr):
    lst = arr.flatten().tolist()
    d = list_of_dicts_to_dicts_of_ndarrays(lst)
    for key, val in d.items():
        d[key] = np.reshape(d[key], arr.shape)
    return d


def dicts_of_lists_to_lists_of_dicts(dictionary):
    return [dict(zip(dictionary, t)) for t in zip(*dictionary.values())]


def ensure_list(arg):
    if not (isinstance(arg, list)):
        try:  # if iterable
            if isinstance(arg, (string_types, dict)):
                arg = [arg]
            elif hasattr(arg, "__iter__"):
                arg = list(arg)
            else:  # if not iterable
                arg = [arg]
        except:  # if not iterable
            arg = [arg]
    return arg


def ensure_string(arg):
    if not (isinstance(arg, string_types)):
        if arg is None:
            return ""
        else:
            return ensure_list(arg)[0]
    else:
        return arg


def flatten_list(lin, sort=False, recursive=False):
    lout = []
    for sublist in lin:
        if recursive and isinstance(sublist, (list, tuple)):
            temp = flatten_list(list(sublist))
        else:
            temp = [sublist]
        for item in temp:
            lout.append(item)
    if sort:
        lout.sort()
    return lout


def flatten_tuple(t, sort=False, recursive=True):
    return tuple(flatten_list(list(t), sort, recursive))


def extract_integer_intervals(iterable, print=False):
    def generator(iterable):
        iterable = sorted(set(iterable))
        for key, group in itertools.groupby(enumerate(iterable),
                                            lambda t: t[1] - t[0]):
            group = list(group)
            yield [group[0][1], group[-1][1]]

    if print:
        output = ""
        for element in generator(iterable):
            if element[0] == element[1]:
                output += "%d, " % element[0]
            else:
                output += "%d...%d, " % tuple(element)
        output = output[:-2]
    else:
        output = []
        for element in generator(iterable):
            if element[0] == element[1]:
                output.append([element[0]])
            else:
                output.append(element)
    return output


def set_list_item_by_reference_safely(ind, item, lst):
    while ind >= len(lst):
        lst.append(None)
    lst.__setitem__(ind, item)


def get_list_or_tuple_item_safely(obj, key):
    try:
        return obj[int(key)]
    except:
        return None


def delete_list_items_by_indices(lin, inds, start_ind=0):
    lout = []
    for ind, l in enumerate(lin):
        if ind + start_ind not in inds:
            lout.append(l)
    return lout


def delete_list_items_by_values(lin, values):
    lout = []
    for l in lin:
        if l not in values:
            lout.append(l)
    return lout


def rotate_n_list_elements(lst, n):
    lst = ensure_list(lst)
    n_lst = len(lst)
    if n_lst != n and n_lst != 0:
        if n_lst == 1:
            lst *= n
        elif n_lst > n:
            lst = lst[:n]
        else:
            old_lst = list(lst)
            while n_lst < n:
                lst += old_lst[0]
                old_lst = old_lst[1:] + old_lst[:1]
    return lst


def linear_index_to_coordinate_tuples(linear_index, shape):
    if len(linear_index) > 0:
        coordinates_tuple = np.unravel_index(linear_index, shape)
        return zip(*[ca.flatten().tolist() for ca in coordinates_tuple])
    else:
        return []


def find_labels_inds(labels, keys, modefun="find", two_way_search=False, break_after=np.iinfo(np.int64).max):
    if isequal_string(modefun, "equal"):
        modefun = lambda x, y: isequal_string(x, y)
    else:
        if two_way_search:
            modefun = lambda x, y: (x.find(y) >= 0) or (y.find(x) >= 0)
        else:
            modefun = lambda x, y: x.find(y) >= 0
    inds = []
    keys = ensure_list(keys)
    labels = ensure_list(labels)
    counts = 0
    for key in keys:
        for label in labels:
            if modefun(label, key):
                inds.append(labels.index(label))
                counts += 1
            if counts >= break_after:
                return inds
    return inds


def extract_dict_stringkeys(d, keys, modefun="find", two_way_search=False,
                            break_after=CalculusConfig.MAX_INT_VALUE, remove=False):
    # TODO: test that it works after modifying with find_labels_inds
    if remove:
        out_dict = deepcopy(d)
    else:
        out_dict = {}
    inds_found = find_labels_inds(d.keys(), keys, modefun, two_way_search, break_after)
    for ikey, (key, value) in enumerate(d.items()):
        if ikey in inds_found:
            if remove:
                del out_dict[key]
            else:
                out_dict.update({key: value})
    return out_dict


def get_val_key_for_first_keymatch_in_dict(name, pkeys, **kwargs):
    pkeys += ["_".join([name, pkey]) for pkey in pkeys]
    temp = extract_dict_stringkeys(kwargs, pkeys, modefun="equal", break_after=1)
    if len(temp) > 0:
        return temp.values()[0], temp.keys()[0].split("_")[-1]
    else:
        return None, None


def labels_to_inds(labels, target_labels):
    if isinstance(target_labels, string_types):
        return_single_element = True
        target_labels = ensure_list(target_labels)
    else:
        target_labels = list(target_labels)
        return_single_element = False
    inds = []
    for lbl in target_labels:
        inds.append(labels.index(lbl))
    if return_single_element:
        # if there was only one label string input
        return inds[0]
    else:
        return inds


def generate_region_labels(n_regions, labels=[], str=". ", numbering=True, numbers=[]):
    if len(numbers) != n_regions:
        numbers = list(range(n_regions))
    if len(labels) == n_regions:
        if numbering:
            return np.array([str.join(["%d", "%s"]) % tuple(l) for l in zip(numbers, labels)])
        else:
            return np.array(labels)
    else:
        return np.array(["%d" % l for l in numbers])


def monopolar_to_bipolar(labels, indices=None, data=None):
    if indices is None:
        indices = range(len(labels))
    bipolar_lbls = []
    bipolar_inds = [[], []]
    for ind in range(len(indices) - 1):
        iS1 = indices[ind]
        iS2 = indices[ind + 1]
        if (labels[iS1][0] == labels[iS2][0]) and \
                int(re.findall(r'\d+', labels[iS1])[0]) == \
                int(re.findall(r'\d+', labels[iS2])[0]) - 1:
            bipolar_lbls.append(labels[iS1] + "-" + labels[iS2])
            bipolar_inds[0].append(iS1)
            bipolar_inds[1].append(iS2)
    if isinstance(data, np.ndarray):
        data = data[bipolar_inds[0]] - data[bipolar_inds[1]]
        return bipolar_lbls, bipolar_inds, data
    else:
        return bipolar_lbls, bipolar_inds


def where(condition, one, two):
    if np.any(condition):
        return one
    else:
        return two


# This function is meant to confirm that two objects assumingly of the same type are equal, i.e., identical
def assert_equal_objects(obj1, obj2, attributes_dict=None, logger=None):
    def print_not_equal_message(attr, field1, field2, logger):
        # logger.error("\n\nValueError: Original and read object field "+ attr + " not equal!")
        # raise_value_error("\n\nOriginal and read object field " + attr + " not equal!")
        warning("Original and read object field " + attr + " not equal!" +
                "\nOriginal field:\n" + str(field1) +
                "\nRead object field:\n" + str(field2), logger)

    if isinstance(obj1, dict):
        get_field1 = lambda obj, key: obj[key]
        if not (isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj1.keys():
                attributes_dict.update({key: key})
    elif isinstance(obj1, (list, tuple)):
        get_field1 = lambda obj, key: get_list_or_tuple_item_safely(obj, key)
        indices = range(len(obj1))
        attributes_dict = dict(zip([str(ind) for ind in indices], indices))
    else:
        get_field1 = lambda obj, attribute: getattr(obj, attribute)
        if not (isinstance(attributes_dict, dict)):
            attributes_dict = dict()
            for key in obj1.__dict__.keys():
                attributes_dict.update({key: key})
    if isinstance(obj2, dict):
        get_field2 = lambda obj, key: obj.get(key, None)
    elif isinstance(obj2, (list, tuple)):
        get_field2 = lambda obj, key: get_list_or_tuple_item_safely(obj, key)
    else:
        get_field2 = lambda obj, attribute: getattr(obj, attribute, None)

    equal = True
    for attribute in attributes_dict:
        # print attributes_dict[attribute]
        field1 = get_field1(obj1, attributes_dict[attribute])
        field2 = get_field2(obj2, attributes_dict[attribute])
        try:
            # TODO: a better hack for the stupid case of an ndarray of a string, such as model.zmode or pmode
            # For non numeric types
            if isinstance(field1, string_types) or isinstance(field1, list) or isinstance(field1, dict) \
                    or (isinstance(field1, np.ndarray) and field1.dtype.kind in 'OSU'):
                if np.any(field1 != field2):
                    print_not_equal_message(attributes_dict[attribute], field1, field2, logger)
                    equal = False
            # For numeric numpy arrays:
            elif isinstance(field1, np.ndarray) and not field1.dtype.kind in 'OSU':
                # TODO: handle better accuracy differences, empty matrices and complex numbers...
                if field1.shape != field2.shape:
                    print_not_equal_message(attributes_dict[attribute], field1, field2, logger)
                    equal = False
                elif np.any(np.float32(field1) - np.float32(field2) > 0):
                    print_not_equal_message(attributes_dict[attribute], field1, field2, logger)
                    equal = False
            # For numeric scalar types
            elif is_numeric(field1):
                if np.float32(field1) - np.float32(field2) > 0:
                    print_not_equal_message(attributes_dict[attribute], field1, field2, logger)
                    equal = False
            else:
                equal = assert_equal_objects(field1, field2, logger=logger)
        except:
            try:
                warning("Comparing str(objects) for field "
                        + str(attributes_dict[attribute]) + " because there was an error!", logger)
                if np.any(str(field1) != str(field2)):
                    print_not_equal_message(attributes_dict[attribute], field1, field2, logger)
                    equal = False
            except:
                raise_value_error("ValueError: Something went wrong when trying to compare "
                                  + str(attributes_dict[attribute]) + " !", logger)

    if equal:
        return True
    else:
        return False


def shape_to_size(shape):
    shape = np.array(shape)
    shape = shape[shape > 0]
    return np.int_(np.max([shape.prod(), 1]))


def shape_to_ndim(shape, squeeze=False):
    if squeeze:
        shape = filter(lambda x: not (np.any(np.in1d(x, [0, 1]))), list(shape))
    return len(shape)


def linspace_broadcast(start, stop, num_steps, maxdims=3):
    x_star = np.linspace(0, 1, num_steps)
    dims = 0
    x = None
    while x is None and dims < maxdims:
        try:
            x = (x_star[:, None] * (stop - start) + start)
        except:
            x_star = x_star[:, np.newaxis]
            dims = dims + 1
    return x


def squeeze_array_to_scalar(arr):
    arr = np.array(arr)
    if arr.size == 1:
        return arr
    elif np.all(arr == arr[0]):
        return arr[0]
    else:
        return arr


def assert_arrays(params, shape=None, transpose=False):
    # type: (object, object, bool) -> object
    if shape is None or \
            not (isinstance(shape, tuple)
                 and len(shape) in range(3) and np.all([isinstance(s, (int, np.int_)) for s in shape])):
        shape = None
        shapes = []  # list of all unique shapes
        n_shapes = []  # list of all unique shapes' frequencies
        size = 0  # initial shape
    else:
        size = shape_to_size(shape)

    for ip in range(len(params)):
        # Convert all accepted types to np arrays:
        if isinstance(params[ip], np.ndarray):
            pass
        elif isinstance(params[ip], (list, tuple)):
            # assuming a list or tuple of symbols...
            params[ip] = np.array(params[ip]).astype(type(params[ip][0]))
        elif is_numeric(params[ip]):
            params[ip] = np.array(params[ip])
        else:
            try:
                import sympy
            except:
                raise_import_error("sympy import failed")
            if isinstance(params[ip], tuple(sympy.core.all_classes)):
                params[ip] = np.array(params[ip])
            else:
                raise_value_error("Input " + str(params[ip]) + " of type " + str(type(params[ip])) + " is not numeric, "
                                                                                                     "of type np.ndarray, nor Symbol")
        if shape is None:
            # Only one size > 1 is acceptable
            if params[ip].size != size:
                if size > 1 and params[ip].size > 1:
                    raise_value_error("Inputs are of at least two distinct sizes > 1")
                elif params[ip].size > size:
                    size = params[ip].size
            # Construct a kind of histogram of all different shapes of the inputs:
            ind = np.array([(x == params[ip].shape) for x in shapes])
            if np.any(ind):
                ind = np.where(ind)[0]
                # TODO: handle this properly
                n_shapes[int(ind)] += 1
            else:
                shapes.append(params[ip].shape)
                n_shapes.append(1)
        else:
            if params[ip].size > size:
                raise_value_error("At least one input is of a greater size than the one given!")

    if shape is None:
        # Keep only shapes of the correct size
        ind = np.array([shape_to_size(s) == size for s in shapes])
        shapes = np.array(shapes)[ind]
        n_shapes = np.array(n_shapes)[ind]
        # Find the most frequent shape
        ind = np.argmax(n_shapes)
        shape = tuple(shapes[ind])

    if transpose and len(shape) > 1:
        if (transpose == "horizontal" or transpose == "row" and shape[0] > shape[1]) or \
                (transpose == "vertical" or transpose == "column" and shape[0] < shape[1]):
            shape = list(shape)
            temp = shape[1]
            shape[1] = shape[0]
            shape[0] = temp
            shape = tuple(shape)

    # Now reshape or tile when necessary
    for ip in range(len(params)):
        try:
            if params[ip].shape != shape:
                if params[ip].size in [0, 1]:
                    params[ip] = np.tile(params[ip], shape)
                else:
                    params[ip] = np.reshape(params[ip], shape)
        except:
            # TODO: maybe make this an explicit message
            logger.info("\n\nWTF?")

    if len(params) == 1:
        return params[0]
    else:
        return tuple(params)


def make_float(x, precision="64"):
    if isinstance(x, np.ndarray):
        if isequal_string(precision, "64"):
            return x.astype(np.float64)
        elif isequal_string(precision, "32"):
            return x.astype(np.float32)
        else:
            return x.astype(np.float_)
    else:
        if isequal_string(precision, "64"):
            return np.float64(x)
        elif isequal_string(precision, "32"):
            np.float32(x)
        else:
            return np.float_(x)


def make_int(x, precision="64"):
    if isinstance(x, np.ndarray):
        if isequal_string(precision, "64"):
            return x.astype(np.int64)
        elif isequal_string(precision, "32"):
            return x.astype(np.int32)
        else:
            return x.astype(np.int_)
    else:
        if isequal_string(precision, "64"):
            return np.int64(x)
        elif isequal_string(precision, "32"):
            np.int32(x)
        else:
            return np.int_(x)


def copy_object_attributes(obj1, obj2, attr1, attr2=None, deep_copy=False, check_none=False):
    attr1 = ensure_list(attr1)
    if attr2 is None:
        attr2 = attr1
    else:
        attr2 = ensure_list(attr2)
    if deep_copy:
        fcopy = lambda a1, a2: setattr(obj2, a2, deepcopy(getattr(obj1, a1)))
    else:
        fcopy = lambda a1, a2: setattr(obj2, a2, getattr(obj1, a1))
    if check_none:
        for a1, a2 in zip(attr1, attr2):
            if getattr(obj2, a2) is None:
                fcopy(a1, a2)
    else:
        for a1, a2 in zip(attr1, attr2):
            fcopy(a1, a2)
    return obj2


def sort_events_by_x_and_y(events, x="senders", y="times",
                           filter_x=None, filter_y=None, exclude_x=[], exclude_y=[], hashfun=str):
    xs = np.array(flatten_list(events[x]))
    if filter_x is None:
        xlabels = np.unique(xs, axis=0).tolist()
    else:
        xlabels = np.unique(flatten_list(filter_x), axis=0).tolist()
    for xlbl in exclude_x:
        try:
            xlabels.remove(xlbl)
        except:
            pass
    ys = flatten_list(events[y])
    if filter_y is not None:
        ys = [yy for yy in ys if yy in flatten_list(filter_y)]
    for yy in exclude_y:
        try:
            ys.remove(yy)
        except:
            pass
    ys = np.array(ys)
    keys = []
    for xlbl in xlabels:
        if not isinstance(xlbl, Hashable):
            keys.append(hashfun(xlbl))
        else:
            keys.append(xlbl)
    if len(ys):
        sorted_events = OrderedDict()
        for key, xlbl in zip(keys, xlabels):
            sorted_events[key] = np.sort(ys[np.where((xs == xlbl).all(axis=-1))])
    else:
        sorted_events = OrderedDict(zip(keys, [np.array([])] * len(keys)))
    return sorted_events


def data_xarray_from_continuous_events(events, times, senders, variables=[],
                                       filter_senders=None, exclude_senders=[], name=None,
                                       dims_names=["Time", "Variable", "Neuron"]):
    unique_times = np.unique(times).tolist()
    if filter_senders is None:
        filter_senders = np.unique(senders).tolist()
    else:
        filter_senders = np.unique(flatten_list(filter_senders)).tolist()
    exclude_senders = ensure_list(exclude_senders)
    for sender in exclude_senders:
        filter_senders.remove(sender)
    variables = ensure_list(variables)
    if len(variables) is None:
        variables = list(events.keys())
    dims_names = ensure_list(dims_names)
    coords = OrderedDict()
    coords[dims_names[0]] = unique_times
    coords[dims_names[1]] = variables
    coords[dims_names[2]] = filter_senders
    n_senders = len(filter_senders)
    n_times = len(unique_times)
    data = np.empty((n_times, len(variables), n_senders))
    last_time = times[0]
    i_time = unique_times.index(last_time)
    i_sender = -1
    for id, (time, sender) in enumerate(zip(times, senders)):
        # Try best guess of next sender:
        i_sender += 1
        if i_sender >= n_senders:
            i_sender = 0
        if filter_senders[i_sender] != sender:
            try:
                i_sender = filter_senders.index(sender)
            except:
                break  # This sender is not one of the chosen filter_senders
        if time != last_time:
            last_time = time
            # Try best guess of next time index:
            i_time += 1
            if i_time >= n_times:
                i_time = n_times - 1
            if time != unique_times[i_time]:
                i_time = unique_times.index(time)
        for i_var, var in enumerate(variables):
            data[i_time, i_var, i_sender] = events[var][id]
    try:
        from xarray import DataArray
        return DataArray(data, dims=list(coords.keys()), coords=coords, name=name)
    except:
        # Return a dictionary as a plan B'
        return {"data": data, "dims": list(coords.keys()), "coords": coords, "name": name}


def concatenate_heterogeneous_DataArrays(data, concat_dim_name,
                                         data_keys=None, name=None, fill_value=np.nan, transpose_dims=None):
    from pandas import Series
    from xarray import concat
    from pandas import Index
    if isinstance(data, (dict, Series)):
        if data_keys is None:
            data_keys = ensure_list(data.keys())
        if isinstance(data, dict):  # dict
            data = ensure_list(data.values())
        else:  # pd.Series
            if name is None:
                name = data.name
            data = ensure_list(data.values)
    data = concat(data, Index(data_keys, name=concat_dim_name), fill_value=fill_value)
    data.name = name
    if transpose_dims:
        data = data.transpose(*transpose_dims)
    return data


def property_to_fun(property):
    if hasattr(property, "__call__"):
        return property
    else:
        return lambda *args, **kwargs: property


def series_loop_generator(ser, inds_or_keys=None):
    index = list(ser.index)
    if inds_or_keys is None:
        inds_or_keys = index
    else:
        inds_or_keys = ensure_list(inds_or_keys)
    for index_or_key in inds_or_keys:
        if isinstance(index_or_key, string_types):
            lbl = index_or_key
            id = index.index(index_or_key)
        else:
            lbl = index[index_or_key]
            id = index_or_key
        pop = ser[index_or_key]
        yield id, lbl, pop
