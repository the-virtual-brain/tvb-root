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


var FourierSpectrum = {
<<<<<<< HEAD
    MARGIN: {top: 30, right: 20, bottom: 30, left: 50},
    matrix_shape: null,
    data_matrix: null,
    x_values_array: null,
    svgContainer: null,
    xAxisScale: null,
    yAxisScale: null,
    xAxis: null,
    yAxis: null,
    xAxisLabel: null,
    yAxisLabel: null,
    plotTitle: null,
    xMin: null,
    xMax: null,
    yMin: null,
    yMax: null,
    url_base: null
};

function Fourier_fourier_spectrum_init(matrix_shape, plotName, xAxisName, yAxisName, x_min, x_max, url_base) {

    FourierSpectrum.xMin = x_min;
    FourierSpectrum.xMax = x_max;

    FourierSpectrum.url_base = url_base;

    FourierSpectrum.matrix_shape = matrix_shape;

    var svgContainer = d3.select("#svg-container");
    FourierSpectrum.svgContainer = svgContainer;

    var xAxisScale = d3.scale.linear();
    var yAxisScale = d3.scale.linear();

    FourierSpectrum.xAxisScale = xAxisScale;
    FourierSpectrum.yAxisScale = yAxisScale;

    FourierSpectrum.xAxis = d3.svg.axis()
        .scale(xAxisScale);
    FourierSpectrum.yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left");

    var xAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(xAxisName);
    var plotTitle = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(plotName);
    var yAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(yAxisName);

    FourierSpectrum.xAxisLabel = xAxisLabel;
    FourierSpectrum.yAxisLabel = yAxisLabel;
    FourierSpectrum.plotTitle = plotTitle;

=======
    matrix_shape: null,
    data_matrix: null,
    x_values_array: null
};

function Fourier_fourier_spectrum_init(matrix_shape, plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id) {

    Plot_plot1d_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id, Fourier_drawDataCurves);
>>>>>>> 9f4b8f98aa160a6382cf1f00fee2700befc8101f

    var x_values_array = [];
    for (var i = 0; i < matrix_shape[0]; i++) {
        x_values_array[i] = ((x_max - x_min) * i) / (matrix_shape[0] - 1) + x_min;
    }
<<<<<<< HEAD
    FourierSpectrum.x_values_array = x_values_array;
}


function Fourier_drawGraph() {
    d3.selectAll("g").remove();
    d3.selectAll("path").remove();
    d3.selectAll("rect").remove();

    var svgContainer = FourierSpectrum.svgContainer;
    var xAxis = FourierSpectrum.xAxis;
    var yAxis = FourierSpectrum.yAxis;
    var xAxisScale = FourierSpectrum.xAxisScale;
    var yAxisScale = FourierSpectrum.yAxisScale;

    var width = svgContainer["0"]["0"].clientWidth - FourierSpectrum.MARGIN.left - FourierSpectrum.MARGIN.right,
        height = svgContainer["0"]["0"].clientHeight - FourierSpectrum.MARGIN.top - FourierSpectrum.MARGIN.bottom;

    xAxisScale.range([FourierSpectrum.MARGIN.left, width - FourierSpectrum.MARGIN.right]).domain([FourierSpectrum.xMin, FourierSpectrum.xMax]);
    yAxisScale.range([height - FourierSpectrum.MARGIN.top, FourierSpectrum.MARGIN.bottom]).domain([FourierSpectrum.yMin, FourierSpectrum.yMax]);
    xAxis.scale(xAxisScale);
    yAxis.scale(yAxisScale);

    svgContainer.append("svg:g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (height - FourierSpectrum.MARGIN.bottom) + ")")
        .call(xAxis);

    svgContainer.append("svg:g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (FourierSpectrum.MARGIN.left) + ",0)")
        .call(yAxis);

    svgContainer.append('rect')
        .attr("height", height - FourierSpectrum.MARGIN.top - FourierSpectrum.MARGIN.bottom)
        .attr("width", width - FourierSpectrum.MARGIN.left - FourierSpectrum.MARGIN.right)
        .attr("transform", "translate(" + (FourierSpectrum.MARGIN.left) + "," + FourierSpectrum.MARGIN.bottom + ")")
        .attr("fill", "white");

    FourierSpectrum.xAxisLabel
        .attr("transform", "translate(" + width / 2 + "," + (height + 10) + ")");
    FourierSpectrum.yAxisLabel
        .attr("transform", "translate(10," + height / 2 + ")" + "rotate(-90)");
    FourierSpectrum.plotTitle
        .attr("transform", "translate(" + width / 2 + ",15)");

    _Fourier_drawDataCurves();
}

function _Fourier_drawDataCurves() {
    var data_matrix = FourierSpectrum.data_matrix;
    var svgContainer = FourierSpectrum.svgContainer;
    var xAxisScale = FourierSpectrum.xAxisScale;
    var yAxisScale = FourierSpectrum.yAxisScale;
    var x_values_array = FourierSpectrum.x_values_array;

    var lineGen = d3.svg.line()
        .x(function (d) {
            return xAxisScale(d[0]);
        })
        .y(function (d) {
            return yAxisScale(d[1]);
        })
        .interpolate("linear");

    for (var i = 0; i < data_matrix.length; i++) {
        var line_data = mergeArrays(x_values_array, data_matrix[i]);
=======
    FourierSpectrum.matrix_shape = matrix_shape;
    FourierSpectrum.x_values_array = x_values_array;
}

function Fourier_drawDataCurves() {
    var svgContainer = Plot1d.svgContainer;
    var data_matrix = FourierSpectrum.data_matrix;
    var x_values_array = FourierSpectrum.x_values_array;
    var lineGen = Plot_drawDataCurves();

    for (var i = 0; i < data_matrix.length; i++) {
        var line_data = _mergeArrays(x_values_array, data_matrix[i]);
>>>>>>> 9f4b8f98aa160a6382cf1f00fee2700befc8101f
        svgContainer.append('svg:path')
            .attr('d', lineGen(line_data))
            .attr('stroke', "#469EEB")
            .attr('stroke-width', 1)
            .attr('fill', 'none');
    }
}

<<<<<<< HEAD
function Fourier_changeXScale(xAxisScale) {
    var svgContainer = FourierSpectrum.svgContainer;
    var width = svgContainer["0"]["0"].clientWidth - FourierSpectrum.MARGIN.left - FourierSpectrum.MARGIN.right;
    var x_min = FourierSpectrum.xMin;
    var x_max = FourierSpectrum.xMax;
    if (xAxisScale === "log")
        FourierSpectrum.xAxisScale = d3.scale.log().range([FourierSpectrum.MARGIN.left, width - FourierSpectrum.MARGIN.right]).domain([x_min, x_max]);
    else
        FourierSpectrum.xAxisScale = d3.scale.linear().range([FourierSpectrum.MARGIN.left, width - FourierSpectrum.MARGIN.right]).domain([x_min, x_max]);
    Fourier_drawGraph();
}

function Fourier_changeYScale(yAxisScale) {
    var svgContainer = FourierSpectrum.svgContainer;
    var height = svgContainer["0"]["0"].clientHeight - FourierSpectrum.MARGIN.top - FourierSpectrum.MARGIN.bottom;
    var y_min = FourierSpectrum.yMin;
    var y_max = FourierSpectrum.yMax;
    if (yAxisScale === "log")
        FourierSpectrum.yAxisScale = d3.scale.log().range([height - FourierSpectrum.MARGIN.top, FourierSpectrum.MARGIN.bottom]).domain([y_min, y_max]);
    else
        FourierSpectrum.yAxisScale = d3.scale.linear().range([height - FourierSpectrum.MARGIN.top, FourierSpectrum.MARGIN.bottom]).domain([y_min, y_max]);
    Fourier_drawGraph();
}

function mergeArrays(array1, array2) {
=======
function _mergeArrays(array1, array2) {
>>>>>>> 9f4b8f98aa160a6382cf1f00fee2700befc8101f
    var result = [];
    for (var i = 0; i < array1.length; i++)
        result[i] = [array1[i], array2[i]];
    return result;
}

function Fourier_changeMode(mode) {
    Fourier_getData($("#state_select option:selected").val(), mode, $("#normalize_select option:selected").text());
}
function Fourier_changeState(state) {
    Fourier_getData(state, $("#mode_select option:selected").text(), $("#normalize_select option:selected").text());
}
function Fourier_changeNormalize(normalized) {
    Fourier_getData($("#state_select option:selected").val(), $("#mode_select option:selected").text(), normalized);
}

function Fourier_getData(state, mode, normalized) {
<<<<<<< HEAD
    let url_base = FourierSpectrum.url_base;
=======
    let url_base = Plot1d.url_base;
>>>>>>> 9f4b8f98aa160a6382cf1f00fee2700befc8101f
    doAjaxCall({
        url: url_base + "selected_state=" + state + ";selected_mode=" + mode + ";normalized=" + normalized,
        type: 'POST',
        async: true,
        success: function (data) {
            data = $.parseJSON(data);
            FourierSpectrum.data_matrix = $.parseJSON(data.data_matrix);
<<<<<<< HEAD
            FourierSpectrum.yMin = data.ymin;
            FourierSpectrum.yMax = data.ymax;
            FourierSpectrum.yAxisScale.domain([data.ymin, data.ymax]);
            Fourier_drawGraph();
=======
            Plot1d.yMin = data.ymin;
            Plot1d.yMax = data.ymax;
            Plot1d.yAxisScale.domain([data.ymin, data.ymax]);
            Plot_drawGraph();
>>>>>>> 9f4b8f98aa160a6382cf1f00fee2700befc8101f
        }
    });
}