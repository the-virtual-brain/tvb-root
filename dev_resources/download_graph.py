# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import csv
import numpy
import matplotlib.pyplot as pyplot


def plot_download_numbers(quarters, dist_count, pypi_count=None, file_name="download_graph.png", is_quarter=True):
    x = range(len(dist_count))
    figure = pyplot.figure(figsize=(15, 7))

    pyplot.plot(x, dist_count, 'b', label="TVB_Distribution", linewidth=3)
    if pypi_count is not None:
        pyplot.plot(x, dist_count + pypi_count, 'g', label="Distribution + Pypi", linestyle="--", linewidth=2)

    pyplot.legend(fontsize=10)
    pyplot.xticks(x, quarters, rotation=25)
    # pyplot.ylim(ymin=0, ymax=3000)
    pyplot.ylabel("Count")
    pyplot.xlabel("Year & " + ("Quarter" if is_quarter else "Month"))
    pyplot.title("TVB Download numbers over the years")
    pyplot.grid(True)

    pyplot.show()
    figure.savefig(file_name)


with open("download_counters.csv") as csv_file:
    rows = list((csv.reader(csv_file)))

col_1 = numpy.array([row[0] + " - " + row[1] if row[1] == "1" else row[1] for row in rows])
col_2 = numpy.array([int(row[2]) for row in rows])
col_3 = numpy.array([int(row[3]) if row[3] != '' else 0 for row in rows])
plot_download_numbers(col_1, col_2, col_3)

with open("download_counters_month.csv") as csv_file:
    rows = list((csv.reader(csv_file)))

col_1 = numpy.array([row[0] + " - " + row[2] if row[2] == "1" else row[2] for row in rows])
col_2 = numpy.array([int(row[3]) for row in rows])
col_3 = numpy.array([int(row[4]) if row[4] != '' else 0 for row in rows])
plot_download_numbers(col_1, col_2, col_3, file_name="download_graph_months.png", is_quarter=False)
