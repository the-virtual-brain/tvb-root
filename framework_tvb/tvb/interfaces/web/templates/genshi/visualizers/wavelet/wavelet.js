/**
 * Created by Dan Pop on 5/24/2017.
 */

function wavelet_spectrogram_view(matrix_data, matrix_shape, start_time, end_time, freq_lo, freq_hi, vmin, vmax) {
    interpolateTerrain = function (t) {
        return t < 0.5 ? i0(t * 2) : i1((t - 0.5) * 2);
    }
    ColSch_initColorSchemeComponent(vmin, vmax);
    ColSch_initColorSchemeGUI(vmin, vmax);
    var data = $.parseJSON(matrix_data);
    var dimensions = $.parseJSON(matrix_shape);
    var n = dimensions[0];
    var m = dimensions[1];
    var canvas = d3.select("canvas")
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
        .attr("transform", "translate(35, " + cHeight + ")")
        .call(xAxis);


    var yAxisScale = d3.scale.linear()
        .domain([freq_lo, freq_hi])
        .range([cHeight, 0]);
    var yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left")
        .ticks(5);
    var yAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(35,0)")
        .call(yAxis);
    context.putImageData(image, 0, 0);
}

function updateLegend(minColor, maxColor) {
    var legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend");
    legendHeight = d3.select("#colorWeightsLegend").node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, "95%");
}
