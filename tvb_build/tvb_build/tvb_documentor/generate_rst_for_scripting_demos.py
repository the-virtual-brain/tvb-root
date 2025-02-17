# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

# I'll admit, I'm just going to copy and paste the output of this into the Demos.rst
# but we should probably eventually maybe make it better.

import os
import os.path
import glob

here = os.path.abspath(os.path.dirname(__file__))
demo_folder = os.path.sep.join([here, '..', '..', '..', 'tvb_documentation', 'demos'])

nburl = 'https://nbviewer.thevirtualbrain.org/url/docs.thevirtualbrain.org/demos'

demos = []
for ipynb_fname in glob.glob(os.path.join(demo_folder, '*.ipynb')):
    _, fname = os.path.split(ipynb_fname)
    title = ' '.join([s.title() for s in fname.split('.')[0].split('_')])
    demos.append((fname, title))

# generate refs first
ref_fmt = '.. _{title}: {nburl}/{fname}'
for fname, title in demos:
    print(ref_fmt.format(fname=fname, title=title, nburl=nburl))

# now figure directives
fig_fmt = '''.. figure:: figures/{bname}.png
      :width: 200px
      :figclass: demo-figure
      :target: `{title}`_

      `{title}`_

'''
for fname, title in demos:
    bname, _ = fname.split('.')
    print(fig_fmt.format(bname=bname, title=title))
