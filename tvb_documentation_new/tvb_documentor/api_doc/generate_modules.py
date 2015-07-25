#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sphinx-autopackage-script

This script parses a directory tree looking for python modules and packages and
creates ReST files appropriately to create code documentation with Sphinx.
It also creates a modules index (named modules.<suffix>).
"""

# Copyright 2008 Société des arts technologiques (SAT), http://www.sat.qc.ca/
# Copyright 2010 Thomas Waldmann <tw AT waldmann-edv DOT de>
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import shutil
import optparse
from tvb_documentor.api_doc.conf import packages_specs, modules_specs

DEFAULT_STYLE = """
.. |tvb| replace:: `The Virtual Brain Project`
.. _tvb: http://www.thevirtualbrain.org

"""

XML_INCLUDE = """
:mod:`%s`
------------------------------

.. automodule:: tvb

.. literalinclude:: %s
    :language: xml
"""

DEMO_INCLUDE = '''
Demo {0}
-----{1}

    .. literalinclude:: {0}
'''

DEMO_TOC = '''
demos
-----

.. toctree::

'''

# automodule options
OPTIONS = ['members',
           'undoc-members',
           # 'inherited-members', # disabled because there's a bug in sphinx
           'show-inheritance']

#skip members on packages with __all__ . Sphinx documents only the entries in
# __all__ and those are usually *not* imported
PACKAGE_WITH_ALL_OPTIONS = [] 

INIT = '__init__.py'

XML_FOLDER = "xml"
AUTHORS_FILE = "AUTHORS"
APPENDABLE_TOCS = ['tvb']

AUTHORS_INCLUDE = """
.. literalinclude:: %s
""" % (os.path.join(XML_FOLDER, AUTHORS_FILE))


def makename(package, module):
    """Join package and module with a dot."""
    # Both package and module can be None/empty.
    if package:
        name = package
        if module:
            name += '.' + module
    else:
        name = module
    return name


def write_file(name, text, opts, extra_toc_entries=""):
    """Write the output file for module/package <name>."""
    if opts.dryrun:
        return
    fname = os.path.join(opts.destdir, "%s.%s" % (name, opts.suffix))
    if os.path.isfile(fname):
        if opts.force and name not in APPENDABLE_TOCS:
            print 'File %s already exists, skipping.' % fname
        elif '.. toctree::' in open(fname).read():
            # Appending extra toc entries only makes sense if we already have a toctree.
            #print '  Appending to file %s.' % fname
            text = open(fname, 'r').read().rstrip()
            start, toc = text.split('.. toctree::')
            new_entries = []
            old_entries = [line for line in toc.split('\n') if line.strip()]
            entries_to_add = [line for line in extra_toc_entries.split('\n') if line.strip()]
            for new_line in entries_to_add:
                if new_line not in old_entries:
                    new_entries.append(new_line)
            text = start + '.. toctree::' + toc + '\n'
            for line in new_entries:
                text += line
                text += '\n'
            f = open(fname, 'w')
            f.write(text)
            f.close()
    else:
        # print '  Creating RST file %s.' % fname
        f = open(fname, 'w')
        f.write(text)
        f.close()


def format_heading(level, text):
    """Create a heading of <level> [1, 2 or 3 supported]."""
    underlining = ['=', '-', '~', ][level - 1] * len(text)
    return '%s\n%s\n\n' % (text, underlining)


def format_directive(module, package=None, options=OPTIONS):
    """Create the automodule directive and add the options."""
    directive = '.. automodule:: %s\n' % makename(package, module)
    for option in options:
        directive += '    :%s:\n' % option
    return directive


def create_module_file(package, module, opts):
    """Build the text of the file and write the file."""
    text = DEFAULT_STYLE
    text += format_heading(1, '%s Module' % module)
    text += format_heading(2, ':mod:`%s` Module' % module)
    text += format_directive(module, package)
    write_file(makename(package, module), text, opts)


def package_has__all_(py_file):
    with open(py_file) as f:
        return '__all__' in f.read()



def create_package_file(root, master_package, subroot, py_files, opts, subs, excludes):
    """Build the text of the file and write the file."""
    package = os.path.split(root)[-1]
    file_name = makename(master_package, subroot)
    if master_package + '.' + subroot in packages_specs:
        return
    # Text that is used in case file already exists and we just want to extend it's toc tree
    extra_toc_entries = "\n"
    text = DEFAULT_STYLE
    text += format_heading(1, ':mod:`%s` Package' % package)
    # add each package's module
    for py_file in py_files:
        py_file_path = os.path.join(root, py_file)
        if shall_skip(py_file_path) or is_excluded(py_file_path, excludes):
            continue
        is_package = py_file == INIT
        py_file_name = os.path.splitext(py_file)[0]
        py_path = makename(subroot, py_file_name)
        #Skip if specific configuration file present and just add to toc tree
        full_module_name = master_package + '.' + (is_package and subroot or py_path)
        if is_package:
            heading = ''
        else:
            heading = ':mod:`%s`' % py_file_name
        if full_module_name in modules_specs:
            shutil.copy(modules_specs[full_module_name], full_module_name + '.rst')
            text += heading + '\n'
            text += '-'.join(['' for _ in xrange(len(heading))])
            text += '\n\n'
            text += '.. toctree::\n'
            text += '    :maxdepth: 2\n\n'
            text += '    %s\n\n' % full_module_name
            continue

        text += format_heading(2, heading)
        if is_package and package_has__all_(py_file_path):
            text += format_directive(is_package and subroot or py_path, master_package,
                                     options=PACKAGE_WITH_ALL_OPTIONS)
        else:
            text += format_directive(is_package and subroot or py_path, master_package)
        text += '\n'

    # build a list of directories that are packages (they contain an INIT file)
    py_subs = [sub for sub in subs if os.path.isfile(os.path.join(root, sub, INIT))]
    xmls = [sub for sub in subs if sub.endswith('.xml')]
    # if there are some package directories, add a TOC for theses subpackages
    if py_subs:
        text += format_heading(2, 'Subpackages')
        text += '.. toctree::\n\n'
        for sub in py_subs:   
            text += '    %s.%s\n' % (makename(master_package, subroot), sub)
            extra_toc_entries += '    %s.%s\n' % (makename(master_package, subroot), sub)
        text += '\n'
    if xmls:
        text += format_heading(2, 'XML resources')
        text += '.. toctree::\n\n'
        for file_n in xmls:   
            text += '   %s\n' % file_n
            extra_toc_entries += '   %s\n' % file_n
    text += '\n'
    extra_toc_entries += '\n'
    
    # Write a new file
    write_file(file_name, text, opts, extra_toc_entries)


def write_verbatim_demos(source, destination):
    def verbatim_rst(f, name):
        content = DEMO_INCLUDE.format(f, len(f) * '-')
        with open(os.path.join(demo_dest, name + '.rst'), 'w') as out:
            out.write(content)
   
    demo_dest = os.path.join(destination, 'demos')
    os.mkdir(demo_dest)
    toc = DEMO_TOC

    for f in os.listdir(source):
        if f.endswith('.py') and f != INIT:
            shutil.copy(os.path.join(source, f), demo_dest)
            name = os.path.splitext(f)[0]
            verbatim_rst(f, name)
            toc += '    demos/' + name
            toc += '\n'

    with open(os.path.join(destination, 'demos.rst'), 'w') as out:
        out.write(toc)


def create_modules_toc_file(master_package, modules, opts, name='index'):
    """
    Create the module's index.
    """
    text = DEFAULT_STYLE
    text += format_heading(1, '%s Modules' % opts.header)
    text += '.. toctree::\n'
    text += '   :maxdepth: %s\n\n' % opts.maxdepth

    modules.sort()
    prev_module = ''
    for module in modules:
        # look if the module is a subpackage and, if yes, ignore it
        if module.startswith(prev_module + '.'):
            continue
        prev_module = module
        text += '   %s\n' % module

    import tvb.simulator
    import tvb.config
    sim_folder = os.path.dirname(tvb.simulator.__file__)

    write_verbatim_demos(
        os.path.join(sim_folder, 'demos'),
        opts.destdir)

    text += '   demos'
    text += '\n'
    text += '\n'    
    frw_folder = os.path.dirname(os.path.dirname(os.path.dirname(tvb.config.__file__)))
    authors_file = os.path.join(frw_folder, AUTHORS_FILE)
    new_file_path = os.path.join(opts.destdir, XML_FOLDER, AUTHORS_FILE)
    shutil.copy(authors_file, new_file_path)
    text += AUTHORS_INCLUDE
    write_file(name, text, opts)


def shall_skip(module):
    """
    Check if we want to skip this module.
    """
    # skip it, if there is nothing (or just \n or \r\n) in the file
    return os.path.getsize(module) < 3


def create_xml_file(package, module, abs_path, opts):
    """
    Create an xml entry for this file.
    """
    text = XML_INCLUDE % (module[0] + module[1], abs_path)
    write_file(makename(package, module[0] + module[1]), text, opts)


def recurse_tree(path, excludes, opts):
    """
    Look for every file in the directory tree and create the corresponding
    ReST files.
    """
    # use absolute path for root, as relative paths like '../../foo' cause
    # 'if "/." in root ...' to filter out *all* modules otherwise
    path = os.path.abspath(path)
    # check if the base directory is a package and get is name
    if INIT in os.listdir(path):
        package_name = path.split(os.path.sep)[-1]
    else:
        package_name = None

    toc = []
    tree = os.walk(path, False)
    for root, subs, files in tree:
        # keep only the Python script files
        py_files = sorted([f for f in files if os.path.splitext(f)[1] == '.py'])
        xml_abs_paths = sorted([os.path.abspath(os.path.join(root, f))
                                for f in files if os.path.splitext(f)[1] == '.xml'])
        xml_files = sorted([f for f in files if os.path.splitext(f)[1] == '.xml'])
        if INIT in py_files:
            py_files.remove(INIT)
            py_files.insert(0, INIT)
        # remove hidden ('.') and private ('_') directories
        subs = sorted([sub for sub in subs if sub[0] not in ['.', '_'] and
                       not is_excluded(os.path.join(root, sub), excludes)])
        # check if there are valid files to process
        if "/." in root or "/_" in root or not py_files or is_excluded(root, excludes):
            continue
        if INIT in py_files:
            # we are in package ...
            # ... with subpackage(s)
            # ... with some module(s)
            # ... with a not-to-be-skipped INIT file
            if subs or len(py_files) > 1 or not shall_skip(os.path.join(root, INIT)):
                subroot = root[len(path):].lstrip(os.path.sep).replace(os.path.sep, '.')
                xml_folder_path = os.path.join(opts.destdir, 'xml')
                if not os.path.exists(xml_folder_path):
                    os.makedirs(xml_folder_path)
                for idx, file_name in enumerate(xml_files):
                    module = os.path.splitext(file_name)         
                    abs_path = xml_abs_paths[idx]
                    new_path = xml_folder_path + os.sep + module[0] + module[1]
                    shutil.copyfile(abs_path, new_path)
                    rst_path = XML_FOLDER + os.sep + module[0] + module[1]
                    create_xml_file(package_name, module, rst_path, opts)
                    subs.append(makename(package_name, module[0] + module[1]))
                create_package_file(root, package_name, subroot, py_files, opts, subs, excludes)
                toc.append(makename(package_name, subroot))
        elif root == path:
            # if we are at the root level, we don't require it to be a package
            for py_file in py_files:
                if not (is_excluded(os.path.join(path, py_file), excludes) or shall_skip(os.path.join(path, py_file))):
                    module = os.path.splitext(py_file)[0]
                    create_module_file(package_name, module, opts)
                    toc.append(makename(package_name, module))

    return package_name, toc


def normalize_excludes(rootpath, excludes):
    """
    Normalize the excluded directory list:
    * must be either an absolute path or start with rootpath,
    * otherwise it is joined with rootpath
    * with trailing slash
    """
    sep = os.path.sep
    f_excludes = []
    for exclude in excludes:
        if not os.path.isabs(exclude) and not exclude.startswith(rootpath):
            exclude = os.path.join(rootpath, exclude)
        if not exclude.endswith(sep):
            exclude += sep
        f_excludes.append(exclude.replace('/', os.path.sep))
    return f_excludes


def is_excluded(root, excludes):
    """
    Check if the directory is in the exclude list.

    Note: by having trailing slashes, we avoid common prefix issues, like
          e.g. an exlude "foo" also accidentally excluding "foobar".
    """
    sep = os.path.sep
    if not root.endswith(sep):
        root += sep
    for exclude in excludes:
        if root.startswith(exclude):
            return True
    return False


def copy_with_replace(module, replacement, source, dest, recurse_lvl=0):
    """
    Copy from source to dest, overwriting files when they already exist in dest.
    """
    for entry in os.listdir(source):
        if entry.startswith('.') or entry.endswith('.sh') or entry.endswith('.py'):
            continue
        full_source_path = os.path.join(source, entry)
        full_dest_path = os.path.join(dest, entry)
        if os.path.isdir(full_source_path):
            if not os.path.isdir(full_dest_path):
                os.makedirs(full_dest_path)
            copy_with_replace(module, replacement, full_source_path, full_dest_path, recurse_lvl + 1)
        else:
            if recurse_lvl == 0 and entry == replacement:
                shutil.copy(full_source_path, os.path.join(dest, module + '.rst'))
            else:
                shutil.copy(full_source_path, full_dest_path)


def process_sources(opts, rootpaths, excludes):
    """
    Initiate generation of RST files for Python code.
        :param opts: execution options dictionary (see below for more details)
        :param rootpaths: root folder where to start parse folders for python modules
        :param excludes: list of folders to be excluded  
    """
    if rootpaths is None:
        raise Exception("Rootpath is required. Please provide a valid folder")
    
    packages = {}
    for rootpath in rootpaths:
        if os.path.isdir(rootpath):
            # check if the output destination is a valid directory
            if opts.destdir and os.path.isdir(opts.destdir):
                normalized_excludes = normalize_excludes(rootpath, excludes)
                package_name, toc = recurse_tree(rootpath, normalized_excludes, opts)
                if package_name in packages:
                    for entry in toc:
                        if entry not in packages[package_name]:
                            packages[package_name].append(entry)
                else:
                    packages[package_name] = toc
            else:
                raise Exception('%s is not a valid output destination directory.' % opts.destdir)
        else:
            raise Exception('%s is not a valid directory.' % rootpath)
    
    for package_name in packages:
        if not opts.notoc:
            create_modules_toc_file(package_name, packages[package_name], opts)
            
    for module in packages_specs:
        details_dict = packages_specs[module]
        copy_with_replace(module, details_dict['main'], details_dict['path'], opts.destdir)


class GenOptions(object):
    def __init__(self, dryrun, suffix, destdir, header, maxdepth, force, notoc):
        self.dryrun = dryrun
        self.suffix = suffix
        self.destdir = destdir
        self.header = header
        self.maxdepth = maxdepth
        self.force = force
        self.notoc = notoc        


def main():
    """
    Parse and check the command line arguments.
    """
    usage_str = ("usage: %prog [options] <package path> [exclude paths, ...]\n" +  
                 "Note: By default this script will not overwrite already created files.")
                
    parser = optparse.OptionParser(usage=usage_str)
    parser.add_option("-n", "--doc-header", action="store", dest="header", help="Documentation Header (default=Project)", default="Project")
    parser.add_option("-d", "--dest-dir", action="store", dest="destdir", help="Output destination directory", default="")
    parser.add_option("-s", "--suffix", action="store", dest="suffix", help="module suffix (default=rst)", default="rst")
    parser.add_option("-m", "--maxdepth", action="store", dest="maxdepth", help="Maximum depth of submodules to show in the TOC (default=4)", type="int", default=4)
    parser.add_option("-r", "--dry-run", action="store_true", dest="dryrun", help="Run the script without creating the files")
    parser.add_option("-f", "--force", action="store_true", dest="force", help="Overwrite all the files")
    parser.add_option("-t", "--no-toc", action="store_true", dest="notoc", help="Don't create the table of content file")
    (opts, args) = parser.parse_args()
    
    if not args:
        parser.error("Package path is required. Please provide it")
    else:
        rootpath, excludes = args[0], args[1:]
        process_sources(opts, [rootpath], excludes)    


if __name__ == '__main__':
    main()

