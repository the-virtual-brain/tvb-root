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

var WaveletSpect = {
    canvas:null,
    data: null,
    n: null,
    m: null,
    xAxisScale: null,
    yAxisScale: null,
    xAxis: null,
    yAxis: null,
    xAxisGroup: null,
    yAxisGroup: null,
    vmin: null,
    vmax: null
};

function wavelet_spectrogram_init(matrix_data, matrix_shape, start_time, end_time, freq_lo, freq_hi, vmin, vmax) {

    ColSch_initColorSchemeComponent(vmin, vmax);
    ColSch_initColorSchemeGUI(vmin, vmax, drawCanvas);

    var data = $.parseJSON(matrix_data);
    var dimensions = $.parseJSON(matrix_shape);
    var n = dimensions[0];
    var m = dimensions[1];
    var canvas = d3.select("canvas")
        .attr("width", m)
        .attr("height", n);

    WaveletSpect.data = data;
    WaveletSpect.n = n;
    WaveletSpect.m = m;
    WaveletSpect.vmin = vmin;
    WaveletSpect.vmax = vmax;
    WaveletSpect.canvas = canvas;

    var context = canvas.node().getContext("2d");
    var cHeight = context.canvas.clientHeight;
    var svgContainer = d3.select("#svg-container");

    var xAxisScale = d3.scale.linear()
        .domain([start_time, end_time]);
    var xAxis = d3.svg.axis()
        .orient("bot")
        .scale(xAxisScale);
    var xAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(35, " + cHeight + ")");
    var yAxisScale = d3.scale.linear()
        .domain([freq_lo, freq_hi]);
    var yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left")
        .ticks(5);
    var yAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(35,0)");

    WaveletSpect.xAxisScale = xAxisScale;
    WaveletSpect.yAxisScale = yAxisScale;
    WaveletSpect.xAxis = xAxis;
    WaveletSpect.yAxis = yAxis;
    WaveletSpect.xAxisGroup = xAxisGroup;
    WaveletSpect.yAxisGroup = yAxisGroup;

    drawCanvas();
    drawAxis(start_time, end_time, freq_lo, freq_hi);
}

function updateLegend(minColor, maxColor) {
    var legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend");
    legendHeight = legendContainer.node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, "95%");
}

function drawCanvas() {
    var data = WaveletSpect.data;
    var n = WaveletSpect.n;
    var m = WaveletSpect.m;
    var vmin = WaveletSpect.vmin;
    var vmax = WaveletSpect.vmax;
    var canvas = WaveletSpect.canvas;
    var context = canvas.node().getContext("2d"),
        image = context.createImageData(m, n);

    for (var i = n - 1; i >= 0; i--) {
        for (var j = 0; j < m; j++) {
            var k = m * i + j;
            var l = (m * (n - i - 1) + j) * 4;
            if (data[k] > vmax)
                data[k] = vmax;
            if (data[k] < vmin)
                data[k] = vmin;
            var c = ColSch_getColor(data[k]);
            image.data[l] = c[0] * 255;
            image.data[l + 1] = c[1] * 255;
            image.data[l + 2] = c[2] * 255;
            image.data[l + 3] = 255;
        }
    }
    context.putImageData(image, 0, 0);
    updateLegend(vmin, vmax);
}

function drawAxis() {
    var canvas = WaveletSpect.canvas;
    var context = canvas.node().getContext("2d");
    var cWidth = context.canvas.clientWidth;
    var cHeight = context.canvas.clientHeight;

    var xAxisScale = WaveletSpect.xAxisScale;
    var yAxisScale = WaveletSpect.yAxisScale;
    xAxisScale.range([0, cWidth]);
    yAxisScale.range([cHeight, 0]);

    var xAxis = WaveletSpect.xAxis.scale(xAxisScale);
    WaveletSpect.xAxisGroup
        .attr("transform", "translate(35, " + cHeight + ")")
        .call(xAxis);

    var yAxis = WaveletSpect.yAxis.scale(yAxisScale);
    WaveletSpect.yAxisGroup
        .attr("transform", "translate(35,0)")
        .call(yAxis);
    updateLegend(WaveletSpect.vmin,WaveletSpect.vmax);
}
