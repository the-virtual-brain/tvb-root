/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

function matrix_view_init_svg(matrix_data, matrix_shape, title, labels, notes, w, h) {
    // setup dimensions, div, svg elements and plotter
    var width = 900;
    var height = 600;
    if (w !== undefined) {
        width = w;
        height = h;
    }

    var div = d3.select("#svg-viewer").attr("style", "width:" + width + "px;");
    var svg = div.append("svg").attr("width", width).attr("height", height);
    var group = svg.append("g").attr("transform", "translate(200, 0)");
    var text = svg.append("g").attr("transform", "translate(20, 100)")
        .append("text").attr("class", "matrix-text");

    var shape = $.parseJSON(matrix_shape);
    labels = $.parseJSON(labels);

    function mat_over(d, i) {
        var x = Math.floor(i / shape[0]);
        var y = Math.floor(i % shape[0]);
        if (labels !== null) {
            x = labels[0][x];
            y = labels[1][y];
        }
        return text.text("M[ " + x + ", " + y + " ] = " + d.toPrecision(3));
    }

    var plot = tv.plot.mat().w(width - 200).h(height).mat_over(mat_over);

    plot.mat(tv.ndar.ndfrom({
            data: $.parseJSON(matrix_data),
            shape: shape
        }
    ));

    plot(group);
    tv.util.usage(div, title, notes);
}
