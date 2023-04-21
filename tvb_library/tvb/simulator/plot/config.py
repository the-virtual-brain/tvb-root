# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
"""
import os
from datetime import datetime

from tvb.basic.profile import TvbProfile


class FiguresConfig(object):
    VERY_LARGE_SIZE = (40, 20)
    VERY_LARGE_PORTRAIT = (30, 50)
    SUPER_LARGE_SIZE = (80, 40)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    NOTEBOOK_SIZE = (20, 10)
    DEFAULT_SIZE = SMALL_SIZE
    FIG_FORMAT = 'png'
    SAVE_FLAG = True
    SHOW_FLAG = False
    MOUSE_HOOVER = False
    WEIGHTS_NORM_PERCENT = 99
    FONTSIZE = 10
    SMALL_FONTSIZE = 8
    LARGE_FONTSIZE = 12
    MAX_VARS_IN_COLS = 3
    MAX_REGIONS_IN_ROWS = 10
    MIN_REGIONS_FOR_RASTER_PLOT = 9

    def largest_size(self):
        import sys
        if 'IPython' not in sys.modules:
            return self.LARGE_SIZE
        from IPython import get_ipython
        if getattr(get_ipython(), 'kernel', None) is not None:
            return self.NOTEBOOK_SIZE
        else:
            return self.LARGE_SIZE

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param out_base: Base folder where figures should be kept
        :param separate_by_run: Set TRUE, when you want figures to be in different files / each run
        """
        self._out_base = out_base or TvbProfile.current.TVB_STORAGE or os.path.join(os.getcwd(), "outputs")
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_FIGURES(self):
        folder = os.path.join(self._out_base, "figs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        return folder


CONFIGURED = FiguresConfig()
