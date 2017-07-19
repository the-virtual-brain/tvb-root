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
 **/


/**
 * Created by Dan Pop on 5/24/2017.
 */

var Pse_isocline = {
    Matrix2d: Matrix2d,
    gid_matrix: null,
    initial_n: null,
    matrix_node_info: null,
    initial_m: null
};

// TODO add hover and click events
function pse_isocline_init(matrix_node_info, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax, gid_matrix, url_base) {

    matrix2d_init(null, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax);
    Pse_isocline.gid_matrix = gid_matrix;
    Pse_isocline.initial_n = Matrix2d.n;
    Pse_isocline.initial_m = Matrix2d.m;
    Pse_isocline.matrix_node_info = matrix_node_info;
    var context = Matrix2d.canvas.node().getContext("2d");
    redrawCanvas(url_base, 'GlobalVariance');

    var canvas = document.getElementById('main-canvas');
    canvas.addEventListener('click', function (evt) {
        var mousePos = getMousePos(canvas, evt);
        displayNodeDetails(getGid(mousePos));
    }, false);

    canvas.addEventListener('mousemove', function (evt) {
        var mousePos = getMousePos(canvas, evt);
        var nodeInfo = getNodeInfo(mousePos);
        var toolTipText = 'x: ' + mousePos.x + '<br/>' + 'y: ' + mousePos.y + "<br/> Operation id: " + nodeInfo["operation_id"] + "<br/> Datatype gid: " + nodeInfo["datatype_gid"] + "<br/> Datatype type: " + nodeInfo["datatype_type"] + "<br/> Datatype subject: " + nodeInfo["datatype_subject"] + "<br/> Datatype invalid: " + nodeInfo["datatype_invalid"];
        var toolTipDiv = d3.select(".matrix2d-toolTip");
        toolTipDiv.html(toolTipText);
        toolTipDiv.style({
            position: "absolute",
            left: mousePos.x + 110 + "px",
            top: mousePos.y + 75 + "px",
            display: "block",
            'background-color': '#C0C0C0',
            border: '1px solid #fdd',
            padding: '2px',
            opacity: 0.80
        })
    }, false);

    canvas.addEventListener('mouseout', function (evt) {
        var toolTipDiv = d3.select(".tooltip");
        toolTipDiv.transition()
            .duration(300)
            .style("display", "none")
    }, false);
}

function redrawCanvas(base_url, selected_metric) {
    doAjaxCall({
        url: base_url + '/' + selected_metric,
        type: 'POST',
        async: false,
        success: function (data) {
            var context = Matrix2d.canvas.node().getContext("2d");
            var dictionar = $.parseJSON(data);
            Matrix2d.data = $.parseJSON(dictionar.matrix_data);
            Matrix2d.vmin = dictionar.vmin;
            Matrix2d.vmax = dictionar.vmax;
            var dimensions = $.parseJSON(dictionar.matrix_shape);
            Matrix2d.n = dimensions[0];
            Matrix2d.m = dimensions[1];
            var interpolatedMatrix = interpolateMatrix(context.canvas.clientWidth, context.canvas.clientHeight);
            Matrix2d.data = matrixToArray(interpolatedMatrix);
            ColSch_initColorSchemeComponent(Matrix2d.vmin, Matrix2d.vmax);
            drawCanvas();
        }
    });
}


function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function getGid(mousePos) {
    var context = Matrix2d.canvas.node().getContext("2d");
    var width = context.canvas.clientWidth;
    var height = context.canvas.clientHeight;
    var i = Math.floor((mousePos.y * Pse_isocline.initial_n) / height);
    var j = Math.floor((mousePos.x * Pse_isocline.initial_m) / width);
    if (i < 0)
        i = 0;
    if (j < 0)
        j = 0;
    return Pse_isocline.gid_matrix[i][j];
}

function getNodeInfo(mousePos) {
    var context = Matrix2d.canvas.node().getContext("2d");
    var width = context.canvas.clientWidth;
    var height = context.canvas.clientHeight;
    var i = Math.floor((mousePos.y * Pse_isocline.initial_n) / height);
    var j = Math.floor((mousePos.x * Pse_isocline.initial_m) / width);
    if (i < 0)
        i = 0;
    if (j < 0)
        j = 0;
    return Pse_isocline.matrix_node_info[i][j];
}