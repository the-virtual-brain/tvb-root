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
    let width = 900;
    let height = 600;
    if (w !== undefined) {
        width = w;
        height = h;
    }

    const div = d3.select("#pearson-viewer").attr("style", "width:" + width + "px;");
    const svg = div.append("svg").attr("width", width).attr("height", height);
    const group = svg.append("g").attr("transform", "translate(200, 0)");
    const text = svg.append("g").attr("transform", "translate(20, 100)")
        .append("text").attr("class", "matrix-text");
    const shape = $.parseJSON(matrix_shape);
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
    PearsonCorrelation.pearson_min = pearson_min;
    PearsonCorrelation.pearson_max = pearson_max;

    tv.util.usage(div, title, notes);
}

function _Pc_plotFunction(matrix_data) {
    d3.selectAll("g").remove();
    const svg = PearsonCorrelation.svg;
    const group = svg.append("g").attr("transform", "translate(200, 0)");
    const text = svg.append("g").attr("transform", "translate(20, 100)")
        .append("text").attr("class", "matrix-text");
    const labels = PearsonCorrelation.labels;
    const shape = PearsonCorrelation.shape;

    function mat_over(d, i) {
        let x = Math.floor(i / shape[0]);
        let y = Math.floor(i % shape[0]);
        if (x < y)
            return "";
        if (labels !== null) {
            x = labels[0][x];
            y = labels[1][y];
        }
        return text.text("M[ " + x + ", " + y + " ] = " + d.toPrecision(3));
    }

    const plot = tv.plot.mat().w(PearsonCorrelation.width - 200).h(PearsonCorrelation.height).mat_over(mat_over);
    plot.half_only(true);
    plot.absolute_min(PearsonCorrelation.pearson_min);
    plot.absolute_max(PearsonCorrelation.pearson_max);

    plot.mat(tv.ndar.ndfrom({
            data: $.parseJSON(matrix_data),
            shape: shape
        }
    ));

    plot(group);
}

function Pc_changeMode(mode) {
    Pc_getData($("#state_select").find("option:selected").val(), mode);
}
function Pc_changeState(state) {
    Pc_getData(state, $("#mode_select").find("option:selected").text());
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