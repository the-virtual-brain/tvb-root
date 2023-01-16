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

import inspect
from importlib import import_module
import tvb.adapters.uploaders
import tvb.adapters.visualizers
import tvb.adapters.datatypes.db
from tvb.adapters.analyzers import ALL_ANALYZERS
from tvb.adapters.creators import ALL_CREATORS
from tvb.adapters.simulator import ALL_SIMULATORS
from tvb.adapters.uploaders import ALL_UPLOADERS
from tvb.adapters.visualizers import ALL_VISUALIZERS
from tvb.adapters.datatypes.db import ALL_DATATYPES
from tvb.config.algorithm_categories import *
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model.model_datatype import DataType
from tvb.adapters.analyzers.metrics_group_timeseries import TimeseriesMetricsAdapter
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.adapters.visualizers.pse_discrete import DiscretePSEAdapter
from tvb.adapters.visualizers.pse_isocline import IsoclinePSEAdapter

LOGGER = get_logger(__name__)


def import_adapters(adapters_top_module, all_adapter_files):
    """
    @:param adapters_top_module: top module under which the ABCAdapter instances are searched for
    @:param all_adapter_files: list of strings representing python submodules. We will import these,
    relative to 'adapters_top_module' and introspect ass concrete subclasses of ABCAdapter defined inside
    @:returns: list of ABCAdapter subclasses
    """
    result = []
    for adapters_file in all_adapter_files:
        try:
            adapters_module = import_module("." + adapters_file, adapters_top_module.__name__)
            for ad_class in dir(adapters_module):
                ad_class = adapters_module.__dict__[ad_class]
                if inspect.isclass(ad_class) and ad_class.__module__ == adapters_module.__name__ and not \
                        inspect.isabstract(ad_class) and issubclass(ad_class, ABCAdapter):
                    if ad_class.can_be_active():
                        result.append(ad_class)
                    else:
                        LOGGER.warning("Skipped Adapter(probably because MATLAB not found):" + str(ad_class))

        except Exception:
            LOGGER.exception("Could not introspect Adapters file:" + adapters_file)
    return result


def import_dt_index(dt_top_module, all_dt_files):
    """
    @:param dt_top_module: top module under which the DataType instances are searched for
    @:param all_dt_files: list of strings representing python submodules. We will import these,
    relative to 'dt_top_module' and introspect ass concrete subclasses of DataType defined inside
    @:returns: list of Dt Index subclasses
    """
    result = []
    for adapters_file in all_dt_files:
        try:
            adapters_module = import_module("." + adapters_file, dt_top_module.__name__)
            for ad_class in dir(adapters_module):
                ad_class = adapters_module.__dict__[ad_class]
                if inspect.isclass(ad_class) and not inspect.isabstract(ad_class) and issubclass(ad_class, DataType):
                    result.append(ad_class)

        except ImportError:
            LOGGER.exception("Could not introspect Adapters file:" + adapters_file)
    return result


class IntrospectionRegistry(object):
    """
    This registry gathers classes that have a role in generating DB tables and rows.
    It is used at introspection time, for the following operations:
        - fill-in all rows in the ALGORITHM_CATEGORIES table
        - fill-in all rows in the ALGORITHMS table
        - generate DB tables for all datatype indexes
        - keep an evidence of the datatype index removers
    All classes that subclass AlgorithmCategoryConfig, ABCAdapter, ABCRemover, HasTraitsIndex should be imported here
    and added to the proper dictionary/list.
    e.g. Each new class of type HasTraitsIndex should be imported here and added to the DATATYPES list.
    """
    ADAPTERS = {
        AnalyzeAlgorithmCategoryConfig: import_adapters(tvb.adapters.analyzers, ALL_ANALYZERS),
        SimulateAlgorithmCategoryConfig: import_adapters(tvb.adapters.simulator, ALL_SIMULATORS),
        UploadAlgorithmCategoryConfig: import_adapters(tvb.adapters.uploaders, ALL_UPLOADERS),
        ViewAlgorithmCategoryConfig: import_adapters(tvb.adapters.visualizers, ALL_VISUALIZERS),
        CreateAlgorithmCategoryConfig: import_adapters(tvb.adapters.creators, ALL_CREATORS),
    }

    DATATYPES = import_dt_index(tvb.adapters.datatypes.db, ALL_DATATYPES)

    SIMULATOR_MODULE = SimulatorAdapter.__module__
    SIMULATOR_CLASS = SimulatorAdapter.__name__

    CONNECTIVITY_MODULE = ConnectivityViewer.__module__
    CONNECTIVITY_CLASS = ConnectivityViewer.__name__

    ALLEN_CREATOR_MODULE = "tvb.adapters.creators.allen_creator"
    ALLEN_CREATOR_CLASS = "AllenConnectomeBuilder"

    SIIBRA_CREATOR_MODULE = "tvb.adapters.creators.siibra_creator"
    SIIBRA_CREATOR_CLASS = "SiibraCreator"

    MEASURE_METRICS_MODULE = TimeseriesMetricsAdapter.__module__
    MEASURE_METRICS_CLASS = TimeseriesMetricsAdapter.__name__

    DISCRETE_PSE_ADAPTER_MODULE = DiscretePSEAdapter.__module__
    DISCRETE_PSE_ADAPTER_CLASS = DiscretePSEAdapter.__name__

    ISOCLINE_PSE_ADAPTER_MODULE = IsoclinePSEAdapter.__module__
    ISOCLINE_PSE_ADAPTER_CLASS = IsoclinePSEAdapter.__name__



