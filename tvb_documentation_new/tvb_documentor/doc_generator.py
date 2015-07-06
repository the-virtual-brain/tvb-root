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
This module is responsible for generation of documentation (pdf, html, ...)
necessary for a distribution.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""
import subprocess
import os
import shutil
import tempfile
from optparse import OptionParser
from sphinx.cmdline import main as sphinx_build

from tvb.basic.logger.builder import get_logger
import tvb_documentor.api_doc.generate_modules
from tvb_documentor.api_doc.generate_modules import process_sources, GenOptions



class DocGenerator:
    """
    This class will generate and copy in the distribution folder documents
    as PDF (into docs folder) and online help (in the web interface static folders).
    """
    USER_GUIDE_FOLDER = "UserGuide"

    # Documents to be processed
    USER_GUIDE = {"name": "UserGuide", "folder": USER_GUIDE_FOLDER}

    USER_GUIDE_UI = {"name": "UserGuide-UI", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_ANALYZE = {"name": "UserGuide-UI_Analyze", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_BURST = {"name": "UserGuide-UI_Simulator", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_CONNECTIVITY = {"name": "UserGuide-UI_Connectivity", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_PROJECT = {"name": "UserGuide-UI_Project", "folder": USER_GUIDE_FOLDER }
    USER_GUIDE_UI_STIMULUS = {"name": "UserGuide-UI_Stimulus", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_USER = {"name": "UserGuide-UI_User", "folder": USER_GUIDE_FOLDER}

    # Folders
    DOCS_SRC = "tvb_documentation"
    DOCS = "docs"
    API = "api"
    DIST_FOLDER = "dist"
    FW_FOLDER = "framework_tvb"
    MANUALS = "manuals"
    ONLINE_HELP = "tvb/interfaces/web/static/help"

    # Documents that will be transformed into PDF
    DOCS_TO_PDF = [USER_GUIDE]

    # Documents that will be transformed into HTML
    DOCS_TO_HTML = [USER_GUIDE_UI, USER_GUIDE_UI_ANALYZE, USER_GUIDE_UI_BURST,
                    USER_GUIDE_UI_CONNECTIVITY, USER_GUIDE_UI_PROJECT,
                    USER_GUIDE_UI_STIMULUS, USER_GUIDE_UI_USER]


    #paths relative to the tvb package that should not be documented
    EXCLUDES = ['simulator/demos', 'simulator/backend', 'simulator/plot',
                'interfaces/web/templates', 'adapters/portlets', 'tests']


    def __init__(self, tvb_root_folder, dist_folder, library_path):
        """
        Creates a new instance.
        :param tvb_root_folder: root tvb folder.
        :param dist_folder: folder where distribution is built.
        :param library_path: folder where TVB code is put into final distribution.
        """
        self.logger = get_logger(self.__class__.__name__)
        self._dist_folder = dist_folder
        self._tvb_root_folder = tvb_root_folder

        # Folders where to store results
        self._dist_docs_folder = os.path.join(self._dist_folder, self.DOCS)
        self._dist_api_folder = os.path.join(self._dist_folder, self.API)
        self._dist_online_help_folder = os.path.join(library_path, self.ONLINE_HELP)

        # Check if folders exist. If not create them
        if not os.path.exists(self._dist_docs_folder):
            os.makedirs(self._dist_docs_folder)
        if os.path.exists(self._dist_online_help_folder):
            shutil.rmtree(self._dist_online_help_folder)
        if not os.path.exists(self._dist_api_folder):
            os.makedirs(self._dist_api_folder)
        os.makedirs(self._dist_online_help_folder)


    def _run_sphinx_online_help(self, dest_folder):

        conf_folder = os.path.dirname(os.path.dirname(tvb_documentor.__file__))

        args = ['anything',  # Ignored but must be there
                '-b', 'html',  # Specify builder: html, dirhtml, singlehtml, txt, latex, pdf,
                '-a',  # Use option "-a" : build all
                '-q',  # Log only Warn and Error

                # these options *override* setting in conf.py
                '-t', 'online_help', # define a tag. It .rst files we can query the build type using this
                '-D', 'templates_path=_templates_online_help', # replace the html layout with a no navigation one
                '-D', 'html_theme_options.nosidebar=True', # do not allocate space for a side bar

                conf_folder,  # Source folder
                dest_folder,  # Output folder
                ]

        status = sphinx_build(args)

        if status != 0:
            raise Exception('sphinx build failure')


    def generate_online_help_via_sphinx(self):
        temp_folder = tempfile.mkdtemp(prefix='tvb_sphinx_rst_online_help_')

        try:
            self._run_sphinx_online_help(temp_folder)

            for doc in self.DOCS_TO_HTML:
                sphinx_root_relative_pth = os.path.join(self.MANUALS, doc['folder'], doc['name'] + '.html')
                src = os.path.join(temp_folder, sphinx_root_relative_pth)
                dest = os.path.join(self._dist_online_help_folder, sphinx_root_relative_pth)

                html_doc_dir = os.path.dirname(dest)
                if not os.path.exists(html_doc_dir):
                    os.makedirs(html_doc_dir)

                shutil.copy(src, dest)

            # We have to copy now necessary styles and images
            shutil.copytree(os.path.join(temp_folder, '_static'),
                            os.path.join(self._dist_online_help_folder, '_static'))
            shutil.copytree(os.path.join(temp_folder, '_images'),
                            os.path.join(self._dist_online_help_folder, '_images'))
        finally:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)


    def _run_sphinx_latex(self, dest_folder):
        conf_folder = os.path.dirname(os.path.dirname(tvb_documentor.__file__))

        args = ['anything',  # Ignored but must be there
                '-b', 'latex',  # Specify builder: html, dirhtml, singlehtml, txt, latex, pdf,
                '-a',  # Use option "-a" : build all
                '-q',  # Log only Warn and Error
                conf_folder,  # Source folder
                dest_folder,  # Output folder
                ]

        status = sphinx_build(args)

        if status != 0:
            raise Exception('sphinx build failure')


    def generate_pdfs_via_sphinx(self):
        """
        This method generates PDF for all available manuals using sphinx and pdflatex.
        """
        temp_folder = tempfile.mkdtemp(prefix='tvb_sphinx_rst_latex_')

        try:
            self._run_sphinx_latex(temp_folder)

            for doc in self.DOCS_TO_PDF:
                tex = doc['name'] + '.tex'
                # run pdflatex
                subprocess.call(["pdflatex", tex ], cwd=temp_folder)
                # 2nd time to include the index
                subprocess.call(["pdflatex", tex ], cwd=temp_folder)
                # copy the pdf's
                pdf_pth = os.path.join(temp_folder, doc['name'] + '.pdf')
                dest_pth = os.path.join(self._dist_docs_folder, doc['name'] + '.pdf')
                shutil.copy(pdf_pth, dest_pth)
        finally:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)


    def generate_api_doc(self):
        """
        This generates API doc in HTML format and include this into distribution.
        """
        # Create a TMP folder where to store generated RST files.
        temp_folder = tempfile.mkdtemp(prefix='tvb_sphinx_rst_')

        try:
            opts = GenOptions(None, 'rst', temp_folder, 'Project', 10, True, None)

            import tvb
            # Start to create RST files needed for Sphinx to generate final HTML
            process_sources(opts, tvb.__path__, self.EXCLUDES)

            # Now generate HTML doc
            conf_folder = os.path.dirname(tvb_documentor.api_doc.generate_modules.__file__)
            args = ['anything',  # Ignored but must be there
                    '-b', 'html',  # Specify builder: html, dirhtml, singlehtml, txt, latex, pdf,
                    '-a',  # Use option "-a" : build all
                    '-q',  # Log only Warn and Error
                    '-c', conf_folder,  # Specify path where to find conf.py
                    '-d', temp_folder,
                    temp_folder,  # Source folder
                    self._dist_api_folder,  # Output folder
                    ]

            sphinx_build(args)
        finally:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)


def main(options, root_folder):
    abs_dist = os.path.join(root_folder, DocGenerator.DIST_FOLDER)
    if os.path.exists(abs_dist):
        shutil.rmtree(abs_dist)
        os.makedirs(abs_dist)

    generator = DocGenerator(root_folder, abs_dist, os.path.join(root_folder, DocGenerator.FW_FOLDER))

    if options.api_doc_only:
        generator.generate_api_doc()

    if options.pdfs_only:
        generator.generate_pdfs_via_sphinx()

    if options.online_help_only:
        generator.generate_online_help_via_sphinx()

    # if no args generate all

    if not options.api_doc_only and not options.pdfs_only and not options.online_help_only:
        generator.generate_api_doc()
        generator.generate_pdfs_via_sphinx()
        generator.generate_online_help_via_sphinx()


if __name__ == "__main__":
    #By default running this module we generate documentation
    ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.join(os.getcwd())))

    PARSER = OptionParser()
    PARSER.add_option("-a", "--api_doc_only", action="store_true", dest="api_doc_only", help="Generates only API DOC")
    PARSER.add_option("-p", "--pdf_only", action="store_true", dest="pdfs_only", help="Generates only manual PDFs")
    PARSER.add_option("-o", "--online_help_only", action="store_true",
                      dest="online_help_only", help="Generates only Online Help")
    OPTIONS, _ = PARSER.parse_args()
    main(OPTIONS, ROOT_FOLDER)
