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


var Plot1d = {
    MARGIN: {top: 30, right: 20, bottom: 30, left: 50},
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
    url_base: null,
    draw_name: null
};

function Plot_plot1d_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id, draw_name) {

    Plot1d.xMin = x_min;
    Plot1d.xMax = x_max;

    Plot1d.url_base = url_base;

    var svgContainer = d3.select("#" + svg_id);
    Plot1d.svgContainer = svgContainer;

    var xAxisScale = d3.scale.linear();
    var yAxisScale = d3.scale.linear();

    Plot1d.xAxisScale = xAxisScale;
    Plot1d.yAxisScale = yAxisScale;

    Plot1d.xAxis = d3.svg.axis()
        .scale(xAxisScale);
    Plot1d.yAxis = d3.svg.axis()
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

    Plot1d.xAxisLabel = xAxisLabel;
    Plot1d.yAxisLabel = yAxisLabel;
    Plot1d.plotTitle = plotTitle;
    Plot1d.draw_name = draw_name;
}


function Plot_drawGraph() {
    d3.selectAll("g").remove();
    d3.selectAll("path").remove();
    d3.selectAll("rect").remove();

    var svgContainer = Plot1d.svgContainer;
    var xAxis = Plot1d.xAxis;
    var yAxis = Plot1d.yAxis;
    var xAxisScale = Plot1d.xAxisScale;
    var yAxisScale = Plot1d.yAxisScale;

    var width = svgContainer["0"]["0"].clientWidth - Plot1d.MARGIN.left - Plot1d.MARGIN.right,
        height = svgContainer["0"]["0"].clientHeight - Plot1d.MARGIN.top - Plot1d.MARGIN.bottom;

    xAxisScale.range([Plot1d.MARGIN.left, width - Plot1d.MARGIN.right]).domain([Plot1d.xMin, Plot1d.xMax]);
    yAxisScale.range([height - Plot1d.MARGIN.top, Plot1d.MARGIN.bottom]).domain([Plot1d.yMin, Plot1d.yMax]);
    xAxis.scale(xAxisScale);
    yAxis.scale(yAxisScale);

    svgContainer.append("svg:g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (height - Plot1d.MARGIN.bottom) + ")")
        .call(xAxis);

    svgContainer.append("svg:g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (Plot1d.MARGIN.left) + ",0)")
        .call(yAxis);

    svgContainer.append('rect')
        .attr("height", height - Plot1d.MARGIN.top - Plot1d.MARGIN.bottom)
        .attr("width", width - Plot1d.MARGIN.left - Plot1d.MARGIN.right)
        .attr("transform", "translate(" + (Plot1d.MARGIN.left) + "," + Plot1d.MARGIN.bottom + ")")
        .attr("fill", "white");

    Plot1d.xAxisLabel
        .attr("transform", "translate(" + width / 2 + "," + (height + 10) + ")");
    Plot1d.yAxisLabel
        .attr("transform", "translate(10," + height / 2 + ")" + "rotate(-90)");
    Plot1d.plotTitle
        .attr("transform", "translate(" + width / 2 + ",15)");
    Plot1d.draw_name();
}

function Plot_drawDataCurves() {
    var xAxisScale = Plot1d.xAxisScale;
    var yAxisScale = Plot1d.yAxisScale;

    return lineGen = d3.svg.line()
        .x(function (d) {
            return xAxisScale(d[0]);
        })
        .y(function (d) {
            return yAxisScale(d[1]);
        })
        .interpolate("linear");
}

function Plot_changeXScale(xAxisScale) {
    var svgContainer = Plot1d.svgContainer;
    var width = svgContainer["0"]["0"].clientWidth - Plot1d.MARGIN.left - Plot1d.MARGIN.right;
    var x_min = Plot1d.xMin;
    var x_max = Plot1d.xMax;
    if (xAxisScale === "Logarithmic")
        Plot1d.xAxisScale = d3.scale.log().range([Plot1d.MARGIN.left, width - Plot1d.MARGIN.right]).domain([x_min, x_max]);
    else
        Plot1d.xAxisScale = d3.scale.linear().range([Plot1d.MARGIN.left, width - Plot1d.MARGIN.right]).domain([x_min, x_max]);
    Plot_drawGraph();
}

function Plot_changeYScale(yAxisScale) {
    var svgContainer = Plot1d.svgContainer;
    var height = svgContainer["0"]["0"].clientHeight - Plot1d.MARGIN.top - Plot1d.MARGIN.bottom;
    var y_min = Plot1d.yMin;
    var y_max = Plot1d.yMax;
    if (yAxisScale === "Logarithmic")
        Plot1d.yAxisScale = d3.scale.log().range([height - Plot1d.MARGIN.top, Plot1d.MARGIN.bottom]).domain([y_min, y_max]);
    else
        Plot1d.yAxisScale = d3.scale.linear().range([height - Plot1d.MARGIN.top, Plot1d.MARGIN.bottom]).domain([y_min, y_max]);
    Plot_drawGraph();
}