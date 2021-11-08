# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
Launches the common landing page

.. moduleauthor:: Bogdan Valean
"""

import os

from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from tvb.basic.logger.builder import get_logger

LOGGER = get_logger('tvb.interfaces.web.run_landing_page')

CSCS_LINK = os.environ.get('CSCS_LINK', None)
CSCS_LABEL = os.environ.get('CSCS_LABEL', None)
if CSCS_LABEL is None:
    CSCS_LABEL = CSCS_LINK

JUL_LINK = os.environ.get('JUL_LINK', None)
JUL_LABEL = os.environ.get('JUL_LABEL', None)
if JUL_LABEL is None:
    JUL_LABEL = JUL_LINK

if CSCS_LINK is None or JUL_LINK is None:
    raise Exception("Make sure you set CSCS_LINK and JUL_LINK env variables.")

server_port = int(os.environ.get('SERVER_PORT', 8080))
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import tvb.interfaces

    CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(tvb.interfaces.__file__)))
except ImportError:
    pass
TEMPLATE_FOLDER = os.path.join(CURRENT_DIR, 'interfaces', 'web', 'templates', 'landing_page')
STATIC_FOLDER = os.path.join(CURRENT_DIR, 'interfaces', 'web', 'static')

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER, static_url_path='/static')


@app.route('/')
def index():
    return render_template('landing.html',
                           apps=[{'link': CSCS_LINK, 'label': CSCS_LABEL}, {'link': JUL_LINK, 'label': JUL_LABEL}])


if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", server_port), app)
    http_server.serve_forever()
