# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Utility file, to compute number of code lines in TVB project folder.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os

INTROSPECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
IGNORED_LIST = {"__init__.py", "project_metrics.py"}

TVB_LIST = []
TVB_CODE_LINES = 0
TEST_LIST = []
CSS_LIST = []
JS_LIST = []
HTML_LIST = []


def count_code_lines(filename):
    count = 0
    with open(filename) as fd:
        for line in fd:
            parts = line.strip().split()
            if len(parts) == 0 or parts[0].startswith('#'):
                continue
            count += 1
    return count


def count_lines(filename):
    count = 0
    with open(filename) as fd:
        for line in fd:
            count += 1
    return count


for pydir, _, pyfiles in os.walk(INTROSPECT_FOLDER):
    for pyfile in pyfiles:
        totalpath = os.path.join(pydir, pyfile)
        tmp = totalpath.split(INTROSPECT_FOLDER)[1]

        if pyfile.endswith(".py") and pyfile not in IGNORED_LIST \
                and not ('externals' in pydir or 'tvb/tests/framework' in pydir or 'tvb/tests/library' in pydir):

            TVB_LIST.append((count_lines(totalpath), tmp))
            TVB_CODE_LINES += count_code_lines(totalpath)

        elif pyfile.endswith(".py") and pyfile not in IGNORED_LIST and 'externals' not in pydir \
                and ('tvb/tests/framework' in pydir or 'tvb/tests/library' in pydir):

            TEST_LIST.append((count_lines(totalpath), tmp))

        elif pyfile.endswith(".css") and '/static/style' in pydir:
            CSS_LIST.append((count_lines(totalpath), tmp))

        elif pyfile.endswith(".js") and '/static/js' in pydir:
            JS_LIST.append((count_lines(totalpath), tmp))

        elif pyfile.endswith(".html") and '/templates/genshi' in pydir:
            HTML_LIST.append((count_lines(totalpath), tmp))

print("Total: %s lines in %s .py TVB files (%d lines of code)" % (
    sum([x[0] for x in TVB_LIST]), len(TVB_LIST), TVB_CODE_LINES))
print("Total: %s lines in %s .py TEST files" % (sum([x[0] for x in TEST_LIST]), len(TEST_LIST)))
print("Total: %s lines in %s .CSS files" % (sum([x[0] for x in CSS_LIST]), len(CSS_LIST)))
print("Total: %s lines in %s .JS files" % (sum([x[0] for x in JS_LIST]), len(JS_LIST)))
print("Total: %s lines in %s .HTML files" % (sum([x[0] for x in HTML_LIST]), len(HTML_LIST)))

TVB_LIST.extend(TEST_LIST)
TVB_LIST.extend(CSS_LIST)
TVB_LIST.extend(JS_LIST)
TVB_LIST.extend(HTML_LIST)

print("\nTotal: %s lines in %s tvb files (%s)" % (sum([x[0] for x in TVB_LIST]), len(TVB_LIST), INTROSPECT_FOLDER))
