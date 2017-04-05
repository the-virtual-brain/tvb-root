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
XML parser for the accepted packages XML.
"""

import os
import sys
from tvb_build import third_party_licenses
import xml.dom.minidom as minidom
from xml.dom.minidom import Node, Document
from package_finder import parse_tree_structure


CURRENT_FOLDER = third_party_licenses.__path__[0]
XML_NODES = ['full-name', 'project-home', 'version', 'usage', 'license',
             'license-type', 'copyright-notice', 'description']
ELEM_ROOT = 'tvb'
ELEM_NODE = 'dependency'
KEY_ENV = 'env'
KEY_NAME = 'name'
KEY_VALUE = 'value'
KEY_VERSION = 'version'
KEY_LICENSE = 'license'
ATT_VALUE = 'value'
FILE = os.path.join(CURRENT_FOLDER, "packages_accepted.xml")

def dumps(thedict):
    """
    :param thedict: A dictionary of dependency_name: dependency_attributes (name version license etc...)
    :return: a dependecy xml as a string
    """
    doc = Document()
    root_node = doc.createElement(ELEM_ROOT)

    for key in sorted(thedict, key=str.lower):
        data = thedict[key]
        dep_node = doc.createElement(ELEM_NODE)
        dep_node.setAttribute(KEY_ENV, data['env'])
        dep_node.setAttribute(KEY_NAME, key)

        for val in XML_NODES:
            node = doc.createElement(val)
            node.setAttribute(KEY_VALUE, data[val])
            dep_node.appendChild(node)
        root_node.appendChild(dep_node)
    doc.appendChild(root_node)
    return doc.toprettyxml(indent="    ", newl='\n')

def _read_all_attributes(node):
    """
    From an XML node, return the map of all attributes.
    """
    atts = {}
    all_attributes = node.attributes
    for i in xrange(all_attributes.length):
        att = all_attributes.item(i)
        atts[att.name] = str(att.value)
    return atts


def _read_all_elements(node):
    """
    From an XML node, return the map of all element values.
    """
    elems = {}
    for child in node.childNodes:
        if child.nodeType != Node.ELEMENT_NODE:
            continue
        elems[str(child.nodeName)] = str(child.attributes[ATT_VALUE].value)
    return elems

def loads(xml_string):
    """
    :param xml_string : a string containing the dependency xml
    :return: a dictionary of {dependency_name: {attributes_dict}}
    """
    accepted_dependencies = {}
    doc_xml = minidom.parseString(xml_string)

    for child in doc_xml.lastChild.childNodes:
        if child.nodeType != Node.ELEMENT_NODE:
            continue

        dependency_attr = _read_all_attributes(child)
        dep_short_name = str(dependency_attr[KEY_NAME])
        dependency_attr[KEY_NAME] = dep_short_name.upper()
        dependency_attr.update(_read_all_elements(child))
        accepted_dependencies[dep_short_name.lower()] = dependency_attr
    return accepted_dependencies


def merge(xmls):
    """
    Merges two or mode dependency xmls.
    Repeating attributes are gathered in a list
    :param xmls: a map xml_file_name : xml_file_contents
    :returns: a merged xml string
    """
    def _merge_node(dst, src):
        for k, v in src.iteritems():
            if k in dst:
                dst[k].update(v)
            else:
                dst[k] = v

    def _flatten_sets_to_strings(node):
        for k, v in node.iteritems():
            if len(v) > 1:
                node[k] = '[' + ','.join(str(a) for a in node[k]) + ']'
            elif len(v) == 1:
                el = next(iter(v))
                node[k] = str(el)
            else:
                node[k] = ''

    merged = {}

    for file_origin_env, src in xmls.iteritems():
        deps = loads(src)
        for name, dep in deps.iteritems():
            if dep['env'] == 'unused':
                continue
            # use the origin of the xml file as the platform. Mainly because linux & linux64 report the same platform
            dep['env'] = file_origin_env
            set_dep = dict((k, set([v])) for k, v in dep.iteritems())
            if name in merged:
                dest = merged[name]
                _merge_node(dest, set_dep)
            else:
                merged[name] = set_dep

    for k, v in merged.iteritems():
        _flatten_sets_to_strings(merged[k])

    return merged


def write(thedict, path):
    with open(path, 'w') as f:
        f.write(dumps(thedict))

def read(path):
    with open(path) as f:
        return loads(f.read())

def read_default():
    """
    Read all dependencies which are currently listed as accepted for TVB.
    """
    with open(FILE) as f:
        return loads(f.read())


def write_used_on_this_platform(accepted, actual, path):
    """
    From a metadictionary of a model entity create the XML file.
    Sets the env attribute to the current platform or 'unused' in the ``accepted`` dict
    """
    for key in accepted:
        data = accepted[key]
        if key in actual:
            data['env'] = sys.platform
        else:
            data['env'] = 'unused'
    write(accepted, path)

    
# Test case for Windows or Mac
if __name__ == '__main__':
    FILE = "packages_accepted.xml"

    import build_licenses

    ACCEPTED = read_default()
    ROOT = build_licenses.ROOT_MAC
    if sys.platform == 'win32':
        ROOT = build_licenses.ROOT_WINDOWS
    ACTUAL = parse_tree_structure(ROOT)
    write_used_on_this_platform(ACCEPTED, ACTUAL, FILE)

