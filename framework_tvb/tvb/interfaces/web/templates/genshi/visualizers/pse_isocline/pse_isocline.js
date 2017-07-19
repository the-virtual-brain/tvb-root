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
    initial_n: null,
    matrix_node_info: null,
    initial_m: null,
    url_base: null,
    canvas_name: null
};

function pse_isocline_init(canvasName, xAxisName, yAxisName, matrix_shape, x_min, x_max, y_min, y_max, url_base, node_info_url) {

    matrix2d_init(canvasName, xAxisName, yAxisName, null, matrix_shape, x_min, x_max, y_min, y_max, null, null);
    Pse_isocline.initial_n = Matrix2d.n;
    Pse_isocline.initial_m = Matrix2d.m;
    Pse_isocline.url_base = url_base;
    Pse_isocline.canvas_name = canvasName;
    // conflict between continuous and discrete IDs
    Pse_isocline.Matrix2d.viewerType = "ISO";
    loadNodeMatrix(node_info_url, matrix_shape);
    var context = Matrix2d.canvas.node().getContext("2d");
    drawAxis();
    var canvas = document.getElementById('main-canvas');
    canvas.addEventListener('click', function (evt) {
        var mousePos = getMousePos(canvas, evt);
        displayNodeDetails(getGid(mousePos));
    }, false);

    canvas.addEventListener('mousemove', function (evt) {
        var mousePos = getMousePos(canvas, evt);
        var nodeInfo = getNodeInfo(mousePos);
        var toolTipText = "Operation id: " + nodeInfo["operation_id"] + "<br/> Datatype gid: " + nodeInfo["datatype_gid"] + "<br/> Datatype type: " + nodeInfo["datatype_type"] + "<br/> Datatype subject: " + nodeInfo["datatype_subject"] + "<br/> Datatype invalid: " + nodeInfo["datatype_invalid"];
        var toolTipDiv = d3.select(".matrix2d-toolTip");
        var canvasParent = document.getElementById("canvasParent");
        var xOffset = Math.floor((canvasParent.clientWidth * 10) / 100);
        var yOffset = Math.floor((canvasParent.clientHeight * 6) / 100);
        toolTipDiv.html(toolTipText);
        toolTipDiv.style({
            position: "absolute",
            left: mousePos.x + xOffset + "px",
            top: mousePos.y + yOffset + "px",
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

function redrawCanvas(base_url, selected_metric, canvasName) {
    base_url = Pse_isocline.url_base;
    canvasName = Pse_isocline.canvas_name;
    doAjaxCall({
        url: base_url + '/' + selected_metric,
        type: 'POST',
        async: false,
        success: function (data) {
            Matrix2d.canvasTitle.text(canvasName + selected_metric);
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
            ColSch_initColorSchemeGUI(Matrix2d.vmin, Matrix2d.vmax, drawCanvas);
            drawCanvas();
        }
    });
}

function loadNodeMatrix(node_info_url, matrix_shape) {
    doAjaxCall({
        url: node_info_url + '/' + matrix_shape,
        type: 'POST',
        async: false,
        success: function (data) {
            Pse_isocline.matrix_node_info = $.parseJSON(data);
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

function getIndicesForMousePosition(mousePos) {
    var context = Matrix2d.canvas.node().getContext("2d");
    var width = context.canvas.clientWidth;
    var height = context.canvas.clientHeight;
    var i = Math.floor((mousePos.y * Pse_isocline.initial_n) / height);
    var j = Math.floor((mousePos.x * Pse_isocline.initial_m) / width);
    if (i < 0)
        i = 0;
    if (j < 0)
        j = 0;
    return [i, j];
}
function getGid(mousePos) {
    var indices = getIndicesForMousePosition(mousePos);
    var i = indices[0];
    var j = indices[1];
    return Pse_isocline.matrix_node_info[i][j]["datatype_gid"];
}

function getNodeInfo(mousePos) {
    var indices = getIndicesForMousePosition(mousePos);
    var i = indices[0];
    var j = indices[1];
    return Pse_isocline.matrix_node_info[i][j];
}