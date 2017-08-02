/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
 *   CITATION:
 * When using The Virtual Brain for scientific publications, please cite it as follows:
 *
 *   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
 *   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
 *       The Virtual Brain: a simulator of primate brain network dynamics.
 *   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
 *
 * .. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
 **/

var PearsonCorrelation = {
    url_base: null,
    width: null,
    height: null,
    svg: null,
    div: null,
    notes: null,
    labels: null,
    shape: null,
    group: null,
    text: null,
    pearson_min: null,
    pearson_max: null,
    title: null
};

function Pc_init(matrix_shape, title, labels, url_base, notes, pearson_min, pearson_max, w, h) {
    // setup dimensions, div, svg elements and plotter
    var width = 900;
    var height = 600;
    if (w !== undefined) {
        width = w;
        height = h;
    }

    var div = d3.select("#pearson-viewer").attr("style", "width:" + width + "px;");
    var svg = div.append("svg").attr("width", width).attr("height", height);
    var group = svg.append("g").attr("transform", "translate(200, 0)");
    var text = svg.append("g").attr("transform", "translate(20, 100)")
        .append("text").attr("class", "matrix-text");

    var shape = $.parseJSON(matrix_shape);
    labels = $.parseJSON(labels);

    PearsonCorrelation.labels = labels;
    PearsonCorrelation.shape = shape;
    PearsonCorrelation.group = group;
    PearsonCorrelation.text = text;
    PearsonCorrelation.svg = svg;
    PearsonCorrelation.height = height;
    PearsonCorrelation.width = width;
    PearsonCorrelation.div = div;
    PearsonCorrelation.notes = notes;
    PearsonCorrelation.title = title;
    PearsonCorrelation.url_base = url_base;
    PearsonCorrelation.absolute_min = pearson_min;
    PearsonCorrelation.absolute_max = pearson_max;

    tv.util.usage(div, title, notes);
}

function _Pc_plotFunction(matrix_data) {
    var svg = PearsonCorrelation.svg;
    d3.selectAll("g").remove();
    var group = svg.append("g").attr("transform", "translate(200, 0)");
    var text = svg.append("g").attr("transform", "translate(20, 100)")
        .append("text").attr("class", "matrix-text");
    var height = PearsonCorrelation.height;
    var width = PearsonCorrelation.width;
    var labels = PearsonCorrelation.labels;
    var shape = PearsonCorrelation.shape;

    function mat_over(d, i) {
        var x = Math.floor(i / shape[0]);
        var y = Math.floor(i % shape[0]);
        if (x < y)
            return "";
        if (labels !== null) {
            x = labels[0][x];
            y = labels[1][y];
        }
        return text.text("M[ " + x + ", " + y + " ] = " + d.toPrecision(3));
    }

    var plot = tv.plot.mat().w(width - 200).h(height).mat_over(mat_over);
    plot.half_only(true);
    plot.pearson_min(PearsonCorrelation.pearson_min);
    plot.pearson_max(PearsonCorrelation.pearson_max);

    plot.mat(tv.ndar.ndfrom({
            data: $.parseJSON(matrix_data),
            shape: shape
        }
    ));

    plot(group);
}

function Pc_changeMode(mode) {
    Pc_getData($("#state_select option:selected").val(), mode);
}
function Pc_changeState(state) {
    Pc_getData(state, $("#mode_select option:selected").text());
}

function Pc_getData(state, mode) {
    let url_base = PearsonCorrelation.url_base;
    doAjaxCall({
        url: url_base + "selected_state=" + state + ";selected_mode=" + mode,
        type: 'POST',
        async: true,
        success: function (data) {
            _Pc_plotFunction(data);
        }
    });
}