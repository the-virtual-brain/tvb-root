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
 **/


/**
 * Created by Dan Pop on 5/24/2017.
 */

var Pse_isocline = {
    Matrix2d: Matrix2d,
    initial_n: null,
    list_guids: null,
    dict_nodes_info: null,
    initial_m: null,
    url_base: null,
    canvas_name: null
};

function pse_isocline_init(canvasName, xAxisName, yAxisName, matrix_shape, matrix_guids, x_min, x_max, y_min, y_max, url_base, node_info_url) {

    matrix2d_init(canvasName, xAxisName, yAxisName, null, matrix_shape, x_min, x_max, y_min, y_max, null, null);
    Pse_isocline.initial_n = Matrix2d.n;
    Pse_isocline.initial_m = Matrix2d.m;
    Pse_isocline.url_base = url_base;
    Pse_isocline.canvas_name = canvasName;
    Pse_isocline.list_guids = matrix_guids;
    // conflict between continuous and discrete IDs
    Pse_isocline.Matrix2d.viewerType = "ISO";
    loadNodeMatrix(node_info_url);
    drawAxis();
    const canvas = document.getElementById('main-canvas-2d');
    canvas.addEventListener('click', function (evt) {
        const mousePos = getMousePos(canvas, evt);
        displayNodeDetails(getGid(mousePos));
    }, false);

    canvas.addEventListener('mousemove', function (evt) {
        const mousePos = getMousePos(canvas, evt);
        const nodeInfo = getNodeInfo(mousePos);
        const toolTipText = nodeInfo.tooltip;
        const toolTipDiv = d3.select(".matrix2d-toolTip");
        const canvasParent = document.getElementById("canvasParent");
        const xOffset = Math.floor((canvasParent.clientWidth * 10) / 100);
        const yOffset = Math.floor((canvasParent.clientHeight * 6) / 100);
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

    canvas.addEventListener('mouseout', function () {
        const toolTipDiv = d3.select(".tooltip");
        toolTipDiv.transition()
            .duration(300)
            .style("display", "none")
    }, false);
}

function redrawCanvas(selected_metric) {
    let base_url = Pse_isocline.url_base;
    let canvasName = Pse_isocline.canvas_name;
    if (base_url === null) {
        console.warn("We won't redraw canvas because data URL hasn't been initialized yet!");
        return;
    }
    doAjaxCall({
        url: base_url + '/' + selected_metric,
        type: 'POST',
        async: true,
        success: function (data) {
            Matrix2d.canvasTitle.text(canvasName + selected_metric);
            const context = Matrix2d.canvas.node().getContext("2d");
            data = $.parseJSON(data);
            Matrix2d.data = $.parseJSON(data.matrix_data);
            Matrix2d.vmin = data.vmin;
            Matrix2d.vmax = data.vmax;
            const dimensions = $.parseJSON(data.matrix_shape);
            Matrix2d.n = dimensions[0];
            Matrix2d.m = dimensions[1];
            const interpolatedMatrix = interpolateMatrix(context.canvas.clientWidth, context.canvas.clientHeight);
            Matrix2d.data = matrixToArray(interpolatedMatrix);
            ColSch_initColorSchemeGUI(Matrix2d.vmin, Matrix2d.vmax, drawCanvas);
            drawCanvas();
        }
    });
}

function loadNodeMatrix(node_info_url) {
    doAjaxCall({
        url: node_info_url,
        type: 'GET',
        async: false,
        success: function (data) {
            Pse_isocline.dict_nodes_info = $.parseJSON(data);
        }
    });
}

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function getIndicesForMousePosition(mousePos) {
    const context = Matrix2d.canvas.node().getContext("2d");
    const width = context.canvas.clientWidth;
    const height = context.canvas.clientHeight;
    let i = Math.floor((mousePos.y * Pse_isocline.initial_n) / height);
    let j = Math.floor((mousePos.x * Pse_isocline.initial_m) / width);
    if (i < 0)
        i = 0;
    if (j < 0)
        j = 0;
    return [i, j];
}
function getGid(mousePos) {
    const indices = getIndicesForMousePosition(mousePos);
    return Pse_isocline.list_guids[indices[0] * Pse_isocline.initial_m + indices[1]];
}

function getNodeInfo(mousePos) {
    const currentGUID = getGid(mousePos);
    return Pse_isocline.dict_nodes_info[currentGUID];
}

function Isocline_MainDraw(groupGID, divId) {
    $('#' + divId).html('');
    doAjaxCall({
        type: "POST",
        url: '/burst/explore/draw_isocline_exploration/' + groupGID,
        success: function (r) {
            $('#' + divId).html(r);
        },
        error: function () {
            displayMessage("Could not refresh with the new metrics.", "warningMessage");
        }
    });
}