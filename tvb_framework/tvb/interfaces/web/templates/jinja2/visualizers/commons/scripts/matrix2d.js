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

    const dimensions = $.parseJSON(matrix_shape);
    const n = dimensions[0];
    const m = dimensions[1];
    const canvas = d3.select("#main-canvas-2d")
        .attr("width", m)
        .attr("height", n);
    if (matrix_data) {
        Matrix2d.data = $.parseJSON(matrix_data);
        Matrix2d.vmin = vmin;
        Matrix2d.vmax = vmax;
        ColSch_initColorSchemeGUI(vmin, vmax, drawCanvas);
    }
    Matrix2d.n = n;
    Matrix2d.m = m;
    Matrix2d.canvas = canvas;

    const context = canvas.node().getContext("2d");
    const cHeight = context.canvas.clientHeight;
    const xLabelHeight = cHeight + 50;
    const yLabelHeight = cHeight / 2;
    const xLabelWidth = context.canvas.clientWidth / 2 + 90;
    const yLabelWidth = 50;
    let svgContainer = d3.select("#svg-container");
    let xAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + xLabelWidth + ", " + xLabelHeight + ")")
        .text(xAxisName);
    let canvasTitle = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + xLabelWidth + ",15)")
        .text(canvasName);
    let yAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + yLabelWidth + ", " + yLabelHeight + ")rotate(-90)")
        .text(yAxisName);
    let xAxisScale = d3.scale.linear()
        .domain([x_min, x_max]);
    let xAxis = d3.svg.axis()
        .orient("bot")
        .scale(xAxisScale);
    let xAxisGroup = svgContainer.append("g")
        .attr("transform", "translate(90, " + cHeight + ")");
    let yAxisScale = d3.scale.linear()
        .domain([y_min, y_max]);
    let yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left")
        .ticks(5);
    let yAxisGroup = svgContainer.append("g")
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
    let legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend" + viewerName);
    legendHeight = legendContainer.node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend" + viewerName);
    ColSch_updateLegendColors(legendContainer.node(), legendHeight * 95 / 100);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, "95%");
}

function drawCanvas() {

    Matrix2d.canvas.attr("width", Matrix2d.m)
        .attr("height", Matrix2d.n);
    const context = Matrix2d.canvas.node().getContext("2d");
    let image = context.createImageData(Matrix2d.m, Matrix2d.n);
    for (let i = Matrix2d.n - 1; i >= 0; i--) {
        for (let j = 0; j < Matrix2d.m; j++) {
            const k = Matrix2d.m * i + j;
            const l = (Matrix2d.m * (Matrix2d.n - i - 1) + j) * 4;
            if (Matrix2d.data[k] > Matrix2d.vmax)
                Matrix2d.data[k] = Matrix2d.vmax;
            if (Matrix2d.data[k] < Matrix2d.vmin)
                Matrix2d.data[k] = Matrix2d.vmin;
            const c = ColSch_getColor(Matrix2d.data[k]);
            image.data[l] = c[0] * 255;
            image.data[l + 1] = c[1] * 255;
            image.data[l + 2] = c[2] * 255;
            image.data[l + 3] = 255;
        }
    }
    context.putImageData(image, 0, 0);
    updateLegend2D(Matrix2d.vmin, Matrix2d.vmax, Matrix2d.viewerType);
}

function drawAxis() {
    const canvas = Matrix2d.canvas;
    if (canvas === null) {
        console.warn("We won't draw axes because canvas has not yet been drawn!");
        return;
    }
    const context = canvas.node().getContext("2d");
    const cWidth = context.canvas.clientWidth;
    const cHeight = context.canvas.clientHeight;
    const canvasParent = document.getElementById("canvasParent");
    const xOffset = Math.floor((canvasParent.clientWidth * 8) / 100);
    const yOffset = Math.floor((canvasParent.clientHeight * 6) / 100);
    const xLabelHeight = cHeight + 2.5 * yOffset;
    const yLabelHeight = cHeight / 2 + yOffset;
    const xLabelWidth = context.canvas.clientWidth / 2 + xOffset;
    const yLabelWidth = xOffset / 2;

    let xAxisScale = Matrix2d.xAxisScale;
    let yAxisScale = Matrix2d.yAxisScale;
    xAxisScale.range([0, cWidth]);
    yAxisScale.range([cHeight, 0]);

    let xAxis = Matrix2d.xAxis.scale(xAxisScale);
    Matrix2d.xAxisGroup
        .attr("transform", "translate(" + xOffset + ", " + (cHeight + yOffset) + ")")
        .call(xAxis);

    let yAxis = Matrix2d.yAxis.scale(yAxisScale);
    Matrix2d.yAxisGroup
        .attr("transform", "translate(" + xOffset + "," + yOffset + ")")
        .call(yAxis);

    Matrix2d.xAxisLabel
        .attr("transform", "translate(" + xLabelWidth + ", " + xLabelHeight + ")");
    Matrix2d.canvasTitle
        .attr("transform", "translate(" + xLabelWidth + "," + yOffset / 2 + ")");
    Matrix2d.yAxisLabel
        .attr("transform", "translate(" + yLabelWidth + ", " + yLabelHeight + ")rotate(-90)")
}

function interpolateMatrix(cWidth, cHeight) {
    let i;
    let j;
    let dataMatrix = [];
    for (i = 0; i < Matrix2d.n; i++) {
        dataMatrix[i] = [];
        for (j = 0; j < Matrix2d.m; j++) {
            dataMatrix[i][j] = undefined;
        }
    }
    const oldMatrix = Matrix2d.data;
    for (i = 0; i < Matrix2d.n; i++)
        for (j = 0; j < Matrix2d.m; j++)
            dataMatrix[i][j] = oldMatrix[i * Matrix2d.m + j];
    const ratioC = Math.floor((cWidth / 10) / Matrix2d.m);
    const ratioL = Math.floor((cHeight / 10) / Matrix2d.n);
    const spaceC = Math.floor((Matrix2d.m * ratioC) / (Matrix2d.m - 1));
    const spaceL = Math.floor((Matrix2d.n * ratioL) / (Matrix2d.n - 1));
    const n = (spaceL) * (Matrix2d.n - 1) + Matrix2d.n;
    const m = (spaceC) * (Matrix2d.m - 1) + Matrix2d.m;
    let newMatrix = [];
    for (i = 0; i < n; i++) {
        newMatrix[i] = [];
        for (j = 0; j < m; j++) {
            newMatrix[i][j] = undefined;
        }
    }
    for (i = 0; i < Matrix2d.n; i++) {
        const line = i * spaceL;
        let col = 0;
        newMatrix[line][col] = dataMatrix[i][0];
        for (let k = 1; k < Matrix2d.m; k++) {
            col += spaceC;
            newMatrix[line][col] = dataMatrix[i][k];
        }
    }
    for (j = 0; j < Matrix2d.m; j++) {
        let column = [];
        for (i = 0; i < Matrix2d.n; i++) {
            column.push(dataMatrix[i][j])
        }
        const interpolatedColumn = interpolateArray(column, n);
        for (i = 0; i < n; i++) {
            newMatrix[i][j * spaceC] = interpolatedColumn[i];
        }
    }
    for (i = 0; i < n; i++) {
        let intermediateLine = newMatrix[i].filter(function (element) {
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
    let newData = [];
    const springFactor = (data.length - 1) / (fitCount - 1);
    newData[0] = data[0]; // for new allocation
    for (let i = 1; i < fitCount - 1; i++) {
        const tmp = i * springFactor;
        const before = Math.floor(tmp).toFixed();
        const after = Math.ceil(tmp).toFixed();
        const atPoint = tmp - before;
        newData[i] = this.linearInterpolate(data[before], data[after], atPoint);
    }
    newData[fitCount - 1] = data[data.length - 1]; // for new allocation
    return newData;
}

function matrixToArray(matrixData) {
    const n = matrixData.length;
    const m = matrixData[0].length;
    let array = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            array[i * m + j] = matrixData[i][j];
        }
    }
    return array;
}
