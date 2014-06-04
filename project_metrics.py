# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

INTROSPECT_FOLDER = os.getcwd()
IGNORED_LIST = {"__init__.py", "project_metrics.py"}

TVB_LIST = []
TEST_LIST = []
CSS_LIST = []
JS_LIST = []
HTML_LIST = []

for pydir, _, pyfiles in os.walk(INTROSPECT_FOLDER):
    for pyfile in pyfiles:
        if pyfile.endswith(".py") and pyfile not in IGNORED_LIST and not ('externals' in pydir or 'tvb.tests.framework' in pydir
                                                                          or 'tvb.tests.library' in pydir):
            totalpath = os.path.join(pydir, pyfile)
            TVB_LIST.append((len(open(totalpath, "r").read().splitlines()), totalpath.split(INTROSPECT_FOLDER)[1]))
        elif pyfile.endswith(".py") and pyfile not in IGNORED_LIST and 'externals' not in pydir \
                and ('tvb.tests.framework' in pydir or 'tvb.tests.library' in pydir):
            totalpath = os.path.join(pydir, pyfile)
            TEST_LIST.append((len(open(totalpath, "r").read().splitlines()), totalpath.split(INTROSPECT_FOLDER)[1]))
        elif pyfile.endswith(".css") and '/static/style' in pydir:
            totalpath = os.path.join(pydir, pyfile)
            CSS_LIST.append((len(open(totalpath, "r").read().splitlines()), totalpath.split(INTROSPECT_FOLDER)[1]))
        elif pyfile.endswith(".js") and '/static/js' in pydir:
            totalpath = os.path.join(pydir, pyfile)
            JS_LIST.append((len(open(totalpath, "r").read().splitlines()), totalpath.split(INTROSPECT_FOLDER)[1]))
        elif pyfile.endswith(".html") and '/templates/genshi' in pydir:
            totalpath = os.path.join(pydir, pyfile)
            HTML_LIST.append((len(open(totalpath, "r").read().splitlines()), totalpath.split(INTROSPECT_FOLDER)[1]))

print "\nTotal: %s lines in %s .py TVB files" % (sum([x[0] for x in TVB_LIST]), len(TVB_LIST))

print "\nTotal: %s lines in %s .py TEST files" % (sum([x[0] for x in TEST_LIST]), len(TEST_LIST))

print "\nTotal: %s lines in %s .CSS files" % (sum([x[0] for x in CSS_LIST]), len(CSS_LIST))

print "\nTotal: %s lines in %s .JS files" % (sum([x[0] for x in JS_LIST]), len(JS_LIST))

print "\nTotal: %s lines in %s .HTML files" % (sum([x[0] for x in HTML_LIST]), len(HTML_LIST))

TVB_LIST.extend(TEST_LIST)
TVB_LIST.extend(CSS_LIST)
TVB_LIST.extend(JS_LIST)
TVB_LIST.extend(HTML_LIST)

print "\nTotal: %s lines in %s tvb files (%s)" % (sum([x[0] for x in TVB_LIST]), len(TVB_LIST), INTROSPECT_FOLDER)


