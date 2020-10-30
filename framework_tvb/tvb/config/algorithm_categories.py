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

DEFAULTDATASTATE_RAW_DATA = 'RAW_DATA'
DEFAULTDATASTATE_INTERMEDIATE = 'INTERMEDIATE'


class AlgorithmCategoryConfig(object):
    """
    Base class to define defaults for an algorithm category entity.
    To be overiden inside each algorithm category module: analyzers, creators, simulator, uploaders, visualizers.
    Subclasses will be used at introspection time, in order to fill the ALGORITHM_CATEGORIES table with data.
    """
    category_name = None
    rawinput = False
    display = False
    launchable = False
    defaultdatastate = DEFAULTDATASTATE_INTERMEDIATE
    order_nr = 999


class AnalyzeAlgorithmCategoryConfig(AlgorithmCategoryConfig):
    category_name = 'Analyze'
    launchable = True
    order_nr = 1


class CreateAlgorithmCategoryConfig(AlgorithmCategoryConfig):
    category_name = 'Create'
    defaultdatastate = DEFAULTDATASTATE_RAW_DATA
    order_nr = 0


class SimulateAlgorithmCategoryConfig(AlgorithmCategoryConfig):
    category_name = 'Simulate'
    order_nr = 0


class UploadAlgorithmCategoryConfig(AlgorithmCategoryConfig):
    category_name = 'Upload'
    rawinput = True
    defaultdatastate = DEFAULTDATASTATE_RAW_DATA
    order_nr = 999


class ViewAlgorithmCategoryConfig(AlgorithmCategoryConfig):
    category_name = 'View'
    display = True
    launchable = True
    order_nr = 3