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

var Matrix2d = {
    canvas: null,
    data: null,
    n: null,
    m: null,
    xAxisScale: null,
    yAxisScale: null,
    xAxis: null,
    yAxis: null,
    xAxisGroup: null,
    xAxisLabel: null,
    yAxisLabel: null,
    canvasTitle: null,
    yAxisGroup: null,
    vmin: null,
    vmax: null,
    viewerType: ""
};

function matrix2d_init(canvasName, xAxisName, yAxisName, matrix_data, matrix_shape, x_min, x_max, y_min, y_max, vmin, vmax) {

    var dimensions = $.parseJSON(matrix_shape);
    var n = dimensions[0];
    var m = dimensions[1];
    var canvas = d3.select("canvas")
        .attr("width", m)
        .attr("height", n);
    if (matrix_data) {
        Matrix2d.data = $.parseJSON(matrix_data);
        Matrix2d.vmin = vmin;
        Matrix2d.vmax = vmax;
        ColSch_initColorSchemeComponent(vmin, vmax);
        ColSch_initColorSchemeGUI(vmin, vmax, drawCanvas);
    }
    Matrix2d.n = n;
    Matrix2d.m = m;
    Matrix2d.canvas = canvas;

    var context = canvas.node().getContext("2d");
    var cHeight = context.canvas.clientHeight;
    var xLabelHeight = cHeight + 50;
    var yLabelHeight = cHeight / 2;
    var xLabelWidth = context.canvas.clientWidth / 2 + 90;
    var yLabelWidth = 50;
    var svgContainer = d3.select("#svg-container");
    var xAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + xLabelWidth + ", " + xLabelHeight + ")")
        .text(xAxisName);
    var canvasTitle = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + xLabelWidth + ",15)")
        .text(canvasName);
    var yAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + yLabelWidth + ", " + yLabelHeight + ")rotate(-90)")
        .text(yAxisName);
    var xAxisScale = d3.scale.linear()
        .domain([x_min, x_max]);
    var xAxis = d3.svg.axis()
        .orient("bot")
        .scale(xAxisScale);
    var xAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(90, " + cHeight + ")");
    var yAxisScale = d3.scale.linear()
        .domain([y_min, y_max]);
    var yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left")
        .ticks(5);
    var yAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(90,0)");

    Matrix2d.xAxisScale = xAxisScale;
    Matrix2d.yAxisScale = yAxisScale;
    Matrix2d.xAxis = xAxis;
    Matrix2d.yAxis = yAxis;
    Matrix2d.xAxisGroup = xAxisGroup;
    Matrix2d.yAxisGroup = yAxisGroup;
    Matrix2d.xAxisLabel = xAxisLabel;
    Matrix2d.yAxisLabel = yAxisLabel;
    Matrix2d.canvasTitle = canvasTitle;

    if (matrix_data) {
        drawAxis(x_min, x_max, y_min, y_max);
        drawCanvas();
    }

}

function updateLegend2D(minColor, maxColor, viewerName) {
    var legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend"+viewerName);
    legendHeight = legendContainer.node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend"+viewerName);
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, "95%");
}

function drawCanvas() {

    var data = Matrix2d.data;
    var n = Matrix2d.n;
    var m = Matrix2d.m;
    var vmin = Matrix2d.vmin;
    var vmax = Matrix2d.vmax;
    var canvas = Matrix2d.canvas;
    canvas.attr("width", m)
        .attr("height", n);
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
    updateLegend2D(vmin, vmax, Matrix2d.viewerType);
}

function drawAxis() {
    var canvas = Matrix2d.canvas;
    var context = canvas.node().getContext("2d");
    var cWidth = context.canvas.clientWidth;
    var cHeight = context.canvas.clientHeight;
    var canvasParent=document.getElementById("canvasParent");
    var xOffset=Math.floor((canvasParent.clientWidth*8)/100);
    var yOffset=Math.floor((canvasParent.clientHeight*6)/100);
    var xLabelHeight = cHeight + 2.5*yOffset;
    var yLabelHeight = cHeight / 2 + yOffset;
    var xLabelWidth = context.canvas.clientWidth / 2 + xOffset;
    var yLabelWidth = xOffset/2;

    var xAxisScale = Matrix2d.xAxisScale;
    var yAxisScale = Matrix2d.yAxisScale;
    xAxisScale.range([0, cWidth]);
    yAxisScale.range([cHeight, 0]);

    var xAxis = Matrix2d.xAxis.scale(xAxisScale);
    Matrix2d.xAxisGroup
        .attr("transform", "translate("+xOffset+", " + (cHeight + yOffset) + ")")
        .call(xAxis);

    var yAxis = Matrix2d.yAxis.scale(yAxisScale);
    Matrix2d.yAxisGroup
        .attr("transform", "translate("+xOffset+","+yOffset+")")
        .call(yAxis);

    Matrix2d.xAxisLabel
        .attr("transform", "translate(" + xLabelWidth + ", " + xLabelHeight + ")");
    Matrix2d.canvasTitle
        .attr("transform", "translate(" + xLabelWidth + ","+yOffset/2+")");
    Matrix2d.yAxisLabel
        .attr("transform", "translate(" + yLabelWidth + ", " + yLabelHeight + ")rotate(-90)")
}

function interpolateMatrix(cWidth, cHeight) {
    var dataMatrix = [];
    for (var i = 0; i < Matrix2d.n; i++) {
        dataMatrix[i] = [];
        for (var j = 0; j < Matrix2d.m; j++) {
            dataMatrix[i][j] = undefined;
        }
    }
    var oldMatrix = Matrix2d.data;
    for (i = 0; i < Matrix2d.n; i++)
        for (j = 0; j < Matrix2d.m; j++)
            dataMatrix[i][j] = oldMatrix[i * Matrix2d.m + j];
    var ratioC = Math.floor((cWidth / 10) / Matrix2d.m);
    var ratioL = Math.floor((cHeight / 10) / Matrix2d.n);
    var spaceC = Math.floor((Matrix2d.m * ratioC) / (Matrix2d.m - 1));
    var spaceL = Math.floor((Matrix2d.n * ratioL) / (Matrix2d.n - 1));
    var n = (spaceL) * (Matrix2d.n - 1) + Matrix2d.n;
    var m = (spaceC) * (Matrix2d.m - 1) + Matrix2d.m;
    var newMatrix = [];
    for (i = 0; i < n; i++) {
        newMatrix[i] = [];
        for (j = 0; j < m; j++) {
            newMatrix[i][j] = undefined;
        }
    }
    for (i = 0; i < Matrix2d.n; i++) {
        var line = i * spaceL;
        var col = 0;
        newMatrix[line][col] = dataMatrix[i][0];
        for (var k = 1; k < Matrix2d.m; k++) {
            col += spaceC;
            newMatrix[line][col] = dataMatrix[i][k];
        }
    }
    for (j = 0; j < Matrix2d.m; j++) {
        var column = [];
        for (i = 0; i < Matrix2d.n; i++) {
            column.push(dataMatrix[i][j])
        }
        interpolatedColumn = interpolateArray(column, n);
        for (i = 0; i < n; i++) {
            newMatrix[i][j * spaceC] = interpolatedColumn[i];
        }
    }
    for (i = 0; i < n; i++) {
        var intermediateLine = newMatrix[i].filter(function (element) {
            return element !== undefined;
        });
        newMatrix[i] = interpolateArray(intermediateLine, m);
    }
    Matrix2d.n = n;
    Matrix2d.m = m;
    return newMatrix;
}

function linearInterpolate(before, after, atPoint) {
    return before + (after - before) * atPoint;
}

function interpolateArray(data, fitCount) {
    var newData = [];
    var springFactor = (data.length - 1) / (fitCount - 1);
    newData[0] = data[0]; // for new allocation
    for (var i = 1; i < fitCount - 1; i++) {
        var tmp = i * springFactor;
        var before = Math.floor(tmp).toFixed();
        var after = Math.ceil(tmp).toFixed();
        var atPoint = tmp - before;
        newData[i] = this.linearInterpolate(data[before], data[after], atPoint);
    }
    newData[fitCount - 1] = data[data.length - 1]; // for new allocation
    return newData;
}

function matrixToArray(matrixData) {
    var n = matrixData.length;
    var m = matrixData[0].length;
    var array = [];
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            array[i * m + j] = matrixData[i][j];
        }
    }
    return array;
}
