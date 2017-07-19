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
    canvas: null,
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


// TODO rename and reuse wavelet
// TODO get matrix data only when needed maybe use a parameter on drawCanvas function
// TODO investigate D3 interpolation mechanism and decide if it can use a colorScheme or rewrite drawCanvas with variable sizes so the result will be closer to actual one
// TODO add hover and click events
function pse_isocline_init(matrix_data, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax) {

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
    var cWidth = context.canvas.clientWidth;
    var svgContainer = d3.select("#svg-container");

    // WaveletSpect.data = matrixInterpolation(cHeight, cWidth);

    var xAxisScale = d3.scale.linear()
        .domain([x_min, x_max]);
    var xAxis = d3.svg.axis()
        .orient("bot")
        .scale(xAxisScale);
    var xAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(35, " + cHeight + ")");
    var yAxisScale = d3.scale.linear()
        .domain([y_min, y_max]);
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
    drawAxis(x_min, x_max, y_min, y_max);
}

function matrixInterpolation(cHeight, cWidth) {
    var dataMatrix = [];

    for (var i = 0; i < WaveletSpect.n; i++) {
        dataMatrix[i] = [];
        for (var j = 0; j < WaveletSpect.m; j++) {
            dataMatrix[i][j] = undefined;
        }
    }
    var oldMatrix = WaveletSpect.data;
    for (var i = 0; i < WaveletSpect.n; i++)
        for (var j = 0; j < WaveletSpect.m; j++)
            dataMatrix[i][j] = oldMatrix[i * WaveletSpect.m + j];

    console.table(dataMatrix);

    var ratio = Math.floor((cWidth / 10) / WaveletSpect.m);
    ratio = 40;
    var n = WaveletSpect.n * ratio;
    var m = WaveletSpect.m * ratio;

    var newMatrix = [];
    for (i = 0; i < n; i++) {
        newMatrix[i] = [];
        for (j = 0; j < m; j++) {
            newMatrix[i][j] = 0;
        }
    }

    // interpolate lines
    var spaceC = Math.floor((WaveletSpect.m * ratio) / (WaveletSpect.m - 1));
    var spaceL = Math.floor((WaveletSpect.n * ratio) / (WaveletSpect.n - 1));
    for (i = 0; i < WaveletSpect.n; i++) {
        var line = i * spaceL;
        if (line === n) {
            line--;
        }
        var spaceC = Math.floor((WaveletSpect.m * ratio) / (WaveletSpect.m - 1));
        var col=0;
        for (var k = 0; k < WaveletSpect.m - 1; k++) {
            var inter = d3.interpolateNumber(dataMatrix[i][k], dataMatrix[i][k + 1]);
            for (j = 0; j < spaceC; j++) {
                newMatrix[line][col] = inter(j / (spaceC - 1));
                col++;
            }
        }
    }

    var resultArray = [];

    for (i = 0; i < WaveletSpect.n; i++) {
        line=i*spaceL;
        for (j = 0; j < m; j++) {
                if(line===n)
                    line--;
                resultArray[i * m + j] = newMatrix[line][j];
        }
    }
    //
    // interpolate columns

    // for (j = 0; j < WaveletSpect.m; j++) {
    //     var col = j * spaceC;
    //     if (col === m) {
    //         col--;
    //     }
    //     spaceL = Math.floor((WaveletSpect.n * ratio) / (WaveletSpect.n - 1));
    //     var line = 0;
    //     for (var k = 0; k < WaveletSpect.n - 1; k++) {
    //         var inter = d3.interpolateNumber(dataMatrix[k][j], dataMatrix[k + 1][j]);
    //         for (i = 0; i < spaceL; i++) {
    //             newMatrix[line][col] = inter(i / (spaceL - 1));
    //             line++;
    //         }
    //     }
    // }
    // var resultArray = [];
    //
    // for (i = 0; i < n; i++) {
    //     for (j = 0; j < m; j++) {
    //             resultArray[i * m + j] = newMatrix[i][j];
    //     }
    // }
    console.table(newMatrix);
    // WaveletSpect.n=n;
    WaveletSpect.m=m;
    return resultArray;
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
    updateLegend(WaveletSpect.vmin, WaveletSpect.vmax);
}
