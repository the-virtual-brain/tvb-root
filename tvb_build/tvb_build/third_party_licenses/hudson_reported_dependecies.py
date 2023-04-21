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


import os
import json
import urllib
import urllib2
import getpass
import cookielib
from tvb_build.third_party_licenses import deps_xml_parser

BASE_URL = 'https://hudson.codemart.ro/hudson/'
LOGIN_URL = BASE_URL + 'login?from=%2Fhudson%2F'
AUTHENTICATION_URL = BASE_URL + 'j_spring_security_check'

URLS2FETCH = {
    'linux': BASE_URL + 'view/TVB/job/TVB%20-%20Build%20-%20Linux/ws/packages_used.xml/',
    'mac': BASE_URL + 'view/TVB/job/TVB%20-%20Build%20-%20Mac/ws/packages_used.xml',
    'win': BASE_URL + 'view/TVB/job/TVB%20-%20Build%20-%20Windows/ws/packages_used.xml',
}


def prepare_conn():
    # Store the cookies and create an opener that will hold them
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    opener.addheaders = [('User-agent', 'Mozilla/5.0'),
                         ('Content-Type', 'application/x-www-form-urlencoded')]

    # Install our opener (note that this changes the global opener to the one
    # we just made, but you can also just call opener.open() if you want)
    urllib2.install_opener(opener)


def login(user, password):
    auth_payload = {
        'j_username': user,
        'j_password': password,
        'from': '/hudson/'
    }

    auth_payload['json'] = json.dumps(auth_payload)
    auth_payload['Submit'] = 'log in'
    # first req to establish a session
    req = urllib2.Request(LOGIN_URL)
    resp = urllib2.urlopen(req)
    resp.read()

    # login request
    data = urllib.urlencode(auth_payload)
    req = urllib2.Request(AUTHENTICATION_URL, data=data)
    resp = urllib2.urlopen(req)
    resp.read()


def fetch_urls():
    ret = {}
    for name, url in URLS2FETCH.iteritems():
        req = urllib2.Request(url)
        resp = urllib2.urlopen(req)
        ret[name] = resp.read()
    return ret


def write_xmls(xmls, pth):
    for name, content in xmls.iteritems():
        with open(os.path.join(pth, name + '.xml'), 'w') as f:
            f.write(content)


def read_xmls(pth):
    ret = {}
    for fname in os.listdir(pth):
        name, ext = os.path.splitext(fname)
        if ext != '.xml':
            continue
        with open(os.path.join(pth, fname)) as f:
            ret[name] = f.read()
    return ret


def main():
    """
    fetch xml's and merge
    """
    prepare_conn()
    usr = raw_input('hudson user: ')
    pwd = getpass.getpass()
    login(usr, pwd)
    xmls = fetch_urls()
    merged = deps_xml_parser.merge(xmls)
    deps_xml_parser.write(merged, 'used_packages.xml')


if __name__ == '__main__':
    main()
