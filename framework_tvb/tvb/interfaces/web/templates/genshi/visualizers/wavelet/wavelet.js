/**
 * Created by Dan Pop on 5/24/2017.
 */

var WaveletSpect = {
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
    WaveletSpect.data = data;
    WaveletSpect.n = n;
    WaveletSpect.m = m;
    WaveletSpect.vmin = vmin;
    WaveletSpect.vmax = vmax;

    drawCanvas();

    var canvas = d3.select("canvas");
    var context = canvas.node().getContext("2d");
    var cWidth = context.canvas.clientWidth;
    var cHeight = context.canvas.clientHeight;
    var svgContainer = d3.select("#svg-container");
    var xAxisScale = d3.scale.linear()
        .domain([start_time, end_time])
        .range([0, cWidth]);

    var xAxis = d3.svg.axis()
        .orient("bot")
        .scale(xAxisScale);
    var xAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(35, " + cHeight + ")");

    var yAxisScale = d3.scale.linear()
        .domain([freq_lo, freq_hi])
        .range([cHeight, 0]);
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

    drawAxis(start_time, end_time, freq_lo, freq_hi);
}

function drawCanvas() {
    var data = WaveletSpect.data;
    var n = WaveletSpect.n;
    var m = WaveletSpect.m;
    var vmin = WaveletSpect.vmin;
    var vmax = WaveletSpect.vmax;

    var canvas = d3.select("#main-canvas")
        .attr("width", m)
        .attr("height", n);
    var context = canvas.node().getContext("2d"),
        image = context.createImageData(m, n);
    for (var i = n - 1; i >= 0; i--) {
        for (var j = 0; j < m; j++) {
            var k = m * i + j;
            var l = (m * (n - i) + j) * 4;
            if (data[k] > vmax)
                data[k] = vmax;
            if (data[k] < vmin)
                data[k] = vmin;
            var c = ColSch_getColor(data[k]);
            image.data[l + 0] = c[0] * 255;
            image.data[l + 1] = c[1] * 255;
            image.data[l + 2] = c[2] * 255;
            image.data[l + 3] = 255;
        }
    }
    context.putImageData(image, 0, 0);
    updateLegend(vmin, vmax);
}

function updateLegend(minColor, maxColor) {
    var legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend");
    legendHeight = d3.select("#colorWeightsLegend").node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, "95%");
}

function drawAxis() {
    var canvas = d3.select("#main-canvas");
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
}
