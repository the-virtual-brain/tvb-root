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


var TOPO_Data = {
    headsData:[],
    valueMin: null,
    valueMax: null,
};

function TOPO_Init(matrix_data, matrix_shape, vmin, vmax, index) {

    if (matrix_data) {
        var dimensions = $.parseJSON(matrix_shape);
        var canvas = d3.select("#canvas-" + (index+1));
        var head = {
            canvas: canvas,
            data: ($.parseJSON(matrix_data)),
            n: dimensions[0],
            m: dimensions[1]
        }
        TOPO_Data.headsData.push(head);
    }

    if(index == 0) {
        TOPO_Data.valueMin = vmin;
        TOPO_Data.valueMax = vmax;
        ColSch_initColorSchemeGUI(vmin, vmax, drawAllCanvases);
    }

    var text_align = $('.topographic_text_allign');
    var cw = text_align.width();
    text_align.css({
        'height': cw + 'px'
    });

    if (matrix_data) {
        drawCanvas(index);
    }
}

function drawAllCanvases(){
    for(var i =0; i < TOPO_Data.headsData.length; i++){
        drawCanvas(i);
    }
}

function drawCanvas(index) {
    var vmin = TOPO_Data.valueMin;
    var vmax = TOPO_Data.valueMax;
    var n = TOPO_Data.headsData[index].n;
    var m = TOPO_Data.headsData[index].m;
    var data = TOPO_Data.headsData[index].data;
    var canvas = TOPO_Data.headsData[index].canvas;

    canvas.attr("width", m).attr("height", n);
    var context = canvas.node().getContext("2d");
    var image = context.createImageData(m, n);
    for (var i = n - 1; i >= 0; i--) {
        for (var j = 0; j < m; j++) {
            var k = m * i + j;
            var l = (m * (n - i - 1) + j) * 4;
            var c = ColSch_getColor(data[k]);
            image.data[l] = c[0] * 255;
            image.data[l + 1] = c[1] * 255;
            image.data[l + 2] = c[2] * 255;
            if (data[k] === vmin - 1) {
                image.data[l + 3] = 0;
            } else {
                image.data[l + 3] = 255;
            }
        }
    }
    context.putImageData(image, 0, 0);

    // updateLegend2D
    var legendContainer = d3.select("#colorWeightsLegend");
    var legendHeight = legendContainer.node().getBoundingClientRect().height;
    var tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), vmin, vmax, "95%");

    drawContour(data, index);
}

function TOPO_DrawContours() {
    for (var i = 0; i < TOPO_Data.headsData.length; i++) {
        drawContour(TOPO_Data.headsData[i].data, i);
    }
}

function drawContour(data, index) {
    var text_align = $('.topographic_text_allign');
    var cw = text_align.width();
    text_align.css({
        'height': cw + 'px'
    });

    var n = TOPO_Data.headsData[index].n;
    var m = TOPO_Data.headsData[index].m;
    var vmin = TOPO_Data.valueMin;
    var vmax = TOPO_Data.valueMax;
    var svg = d3.select("#svg-container-" + (index+1));
    var width = svg["0"]["0"].clientWidth;
    var height = svg["0"]["0"].clientHeight;
    var scale = 0.8;
    data = flipArray(data, n, m);
    svg.selectAll("path").remove();
    svg.selectAll("path")
        .data(d3.contours()
            .size([n, m])
            .thresholds(d3.range(vmin, vmax, (vmax - vmin) / 10))
            (data))
        .enter().append("path")
        .attr("d", d3.geoPath(d3.geoIdentity().scale((width / n) * scale)))
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("overflow", "hidden")
        .attr("transform", "translate(" + width * (1 - scale) / 2 + "," + (height * 0.15) + ")")
        .attr("fill", "none");
}

function flipArray(array, n, m) {
    var newArray = [];
    for (var i = n - 1; i >= 0; i--) {
        for (var j = 0; j < m; j++) {
            newArray.push(array[i * n + j]);
        }
    }
    return newArray;
}