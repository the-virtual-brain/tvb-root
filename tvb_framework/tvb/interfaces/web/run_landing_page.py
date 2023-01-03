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

"""
Launches the common landing page

.. moduleauthor:: Bogdan Valean
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from flask import Flask, render_template
from gevent.pywsgi import WSGIServer
from tvb.basic.logger.builder import get_logger

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import tvb.interfaces

    CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(tvb.interfaces.__file__)))
except ImportError:
    pass

LOGGER = get_logger('tvb.interfaces.web.run_landing_page')
TEMPLATE_FOLDER = os.path.join(CURRENT_DIR, 'interfaces', 'web', 'templates', 'landing_page')
STATIC_FOLDER = os.path.join(CURRENT_DIR, 'interfaces', 'web', 'static')
REDIRECT_OPTIONS = []
app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
            static_folder=STATIC_FOLDER, static_url_path='/static')


class RedirectOption(object):
    def __init__(self, url, label, description):
        self.url = url
        self.label = label
        self.description = description

    def __repr__(self):
        return "RedirectOption( " + self.label + " , " + self.url + ")"


def parse_config_file(file_name=os.path.expanduser("~/.tvb.landing.page.configuration")):
    result = []
    no_of_redirect_options = "0"
    try:
        config_dict = {}
        with open(file_name, 'r') as cfg_file:
            data = cfg_file.read()
            entries = [line for line in data.split('\n') if not line.startswith('#') and len(line.strip()) > 0]
            for one_entry in entries:
                name, value = one_entry.split('=', 1)
                config_dict[name] = value

        no_of_redirect_options = config_dict.get('no_of_options')
        for i in range(int(no_of_redirect_options)):
            result.append(RedirectOption(config_dict.get(str(i) + ".option.url"),
                                         config_dict.get(str(i) + ".option.label"),
                                         config_dict.get(str(i) + ".option.description")))
    except Exception:
        LOGGER.exception("Could not parse the configuration file:" + str(file_name))

    LOGGER.info("Parsed " + no_of_redirect_options + " redirect options")
    LOGGER.info(result)
    return result


@app.route('/')
def index():
    return render_template('landing.html', apps=REDIRECT_OPTIONS)


if __name__ == '__main__':
    REDIRECT_OPTIONS = parse_config_file()
    server_port = int(os.environ.get('SERVER_PORT', 8080))
    http_server = WSGIServer(("0.0.0.0", server_port), app)
    http_server.serve_forever()
