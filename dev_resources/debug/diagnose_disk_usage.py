# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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

import os
import argparse
from tvb.core.entities.storage import dao
from tvb.file.files_helper import FilesHelper
from tvb.core.entities.model.model_project import Project


class DiagnoseDiskUsage(object):
    FORMAT_DT = '    {:14} {:20} {:>12} {:>12} {:>12} {:>12}'
    HEADER_DT = FORMAT_DT.format('', '', 'disk_size(kib)', 'db_size(kib)', 'delta(kib)', 'ratio(%)')

    def __init__(self, prj_id):
        self.file_helper = FilesHelper()
        self.expected_files = set()
        self.prj_disk_size, self.prj_db_size = 0, 0

        try:
            dao.session.open_session()
            # We do not fetch the project using dao because dao will detach it from the session.
            # We want to query on the fly on attribute access and this requires attached objects.
            # This code is doing a tree traversal of the db.
            # The query on attribute access style fits better than aggregating queries.
            self.project = dao.session.query(Project).filter(Project.id == prj_id).one()
            self.expected_files.add(self.file_helper.get_project_meta_file_path(self.project.name))
            root_path = self.file_helper.get_project_folder(self.project)

            print()
            print('Reporting disk for project {} in {}'.format(self.project.name, root_path))
            print()
            print(self.HEADER_DT)

            for op in self.project.OPERATIONS:
                self.analyse_operation(op)

            print(self.HEADER_DT)
            self.print_usage_line('Project', 'total', self.prj_disk_size, self.prj_db_size)

            print()
            self.list_unexpected_project_files(root_path)
            print()
        finally:
            dao.session.close_session()

    @staticmethod
    def get_h5_by_gid(root, gid):
        for f in os.listdir(root):
            fp = os.path.join(root, f)
            if gid in f and os.path.isfile(fp):
                return fp

    @staticmethod
    def get_file_kib_size(fp):
        return int(round((os.path.getsize(fp) / 1024.)))

    @staticmethod
    def print_usage_line(col1, col2, actual, expected):
        if expected != 0:
            ratio = int(100.0 * actual / expected)
            if ratio > 200:
                ratio = "! %s" % ratio
            else:
                ratio = str(ratio)
        else:
            ratio = 'inf'

        delta = actual - expected
        if delta > 100:
            delta = "! %s" % delta
        else:
            delta = str(delta)

        print(DiagnoseDiskUsage.FORMAT_DT.format(col1, col2, '{:,}'.format(actual),
                                                 '{:,}'.format(expected), delta, ratio))

    def analyse_operation(self, op):
        op_disk_size, op_db_size = 0, 0

        print('Operation {} : {}'.format(op.id, op.algorithm.name))

        for dt in op.DATA_TYPES:
            if dt.type == 'DataTypeGroup':
                # these have no h5
                continue
            op_pth = self.file_helper.get_operation_folder(self.project.name, op.id)
            dt_pth = self.get_h5_by_gid(op_pth, dt.gid)

            dt_actual_disk_size = self.get_file_kib_size(dt_pth)

            db_disk_size = dt.disk_size or 0

            op_disk_size += dt_actual_disk_size
            op_db_size += db_disk_size

            self.print_usage_line(dt.gid[:12], dt.type, dt_actual_disk_size, db_disk_size)
            self.expected_files.add(dt_pth)

        self.prj_disk_size += op_disk_size
        self.prj_db_size += op_db_size
        self.print_usage_line('', 'total :', op_disk_size, op_db_size)
        print()

    def list_unexpected_project_files(self, root_path):
        unexpected = []
        for r, d, files in os.walk(root_path):
            for f in files:
                pth = os.path.join(r, f)
                if pth not in self.expected_files:
                    unexpected.append(pth)

        print('Unexpected project files :')
        for f in unexpected:
            print(f)

        if not unexpected:
            print('yey! none found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse TVB disk usage vs db reported usage.')
    parser.add_argument('project_id', type=str, help='project id')
    args = parser.parse_args()

    DiagnoseDiskUsage(args.project_id)
