# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

# I'll admit, I'm just going to copy and paste the output of this into the Demos.rst
# but we should probably eventually maybe make it better.

import os
import os.path
import glob

here = os.path.abspath(os.path.dirname(__file__))
demo_folder = os.path.sep.join([here, '..', '..', '..', 'tvb_documentation', 'demos'])

nburl = 'https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos'

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
