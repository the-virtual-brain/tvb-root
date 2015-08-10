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
import cookielib
import urllib
import urllib2
import json
import os
import getpass
from tvb_build.third_party_licenses import deps_xml_parser

BASE_URL = 'https://hudson.codemart.ro/hudson/'
LOGIN_URL = BASE_URL + 'login'
AUTHENTICATION_URL = BASE_URL + 'j_acegi_security_check'
URLS2FETCH = {
    'linux64' : BASE_URL + 'view/TVB/job/TVB%20-%20Linux64/ws/packages_used.xml',
    'linux32' : BASE_URL + 'view/TVB/job/TVB%20-%20Linux32/ws/packages_used.xml',
    'macos64' : BASE_URL + 'view/TVB/job/TVB%20-%20MacOS%20x64/ws/packages_used.xml',
    'macos32' : BASE_URL + 'view/TVB/job/TVB%20-%20MacOS/ws/packages_used.xml',
    'win64' : BASE_URL + 'view/TVB/job/TVB%20-%20Windows%20x64/ws/packages_used.xml',
    'win32' : BASE_URL + 'view/TVB/job/TVB%20-%20Windows/ws/packages_used.xml'
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
        'remember_me': 'false',
        'from': '/hudson/',
        'Submit':'log in'
    }

    auth_payload['json'] = json.dumps(auth_payload)
    # first req to establish a session
    req = urllib2.Request(LOGIN_URL)
    resp = urllib2.urlopen(req)
    resp.read()

    # login request
    data = urllib.urlencode(auth_payload)
    req = urllib2.Request(AUTHENTICATION_URL, data)
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

def main_fetch(pth):
    """
    fetch the xml files from the build machines
    """
    prepare_conn()
    usr = raw_input('user: ')
    pwd = getpass.getpass()
    login(usr, pwd)
    xmls = fetch_urls()
    write_xmls(xmls, pth)

def main_merge(pth):
    xmls = read_xmls(pth)
    merged = deps_xml_parser.merge(xmls)
    deps_xml_parser.write(merged, os.path.join(pth, 'used_packages.xml'))

def main():
    """
    fetch xml's and merge
    """
    prepare_conn()
    usr = raw_input('user: ')
    pwd = getpass.getpass()
    login(usr, pwd)
    xmls = fetch_urls()
    merged = deps_xml_parser.merge(xmls)
    deps_xml_parser.write(merged, 'used_packages.xml')

if __name__ == '__main__':
    main()
