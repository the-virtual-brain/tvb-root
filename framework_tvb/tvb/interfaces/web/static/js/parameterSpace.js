/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 as published by the Free
 * Software Foundation. This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details. You should have received a copy of the GNU General
 * Public License along with this program; if not, you can download it here
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
 *
 **/
/* global doAjaxCall, displayMessage */
//general chores
//todo create an exporting function that can save the figure
//todo create a function that will normalize the size attributes that belong to the results


//finish creating the explore box, and it's little tooltip window.
//should the hover tooltip have information about the color and weight metric?

// what to start on tomorrow: try to determine the relation to the correct values for the brush section and the placement of the grids.
//also write a grid function that is flexible to input stepvalue and doesn't have a base line to make things look wierd
//change the selection bars to being only session stored, and make them not specific to the datatypegid


// We keep all-nodes information for current PSE as a global, to have them ready at node-selection, node-overlay.
var PSE_nodesInfo;
// Keep Plot-options and MIN/MAx colors for redraw (e.g. at resize).
var _PSE_plotOptions;
var _PSE_minColor;
var _PSE_maxColor;
var _PSE_plot;

/*
 * @param canvasId: the id of the HTML DIV on which the drawing is done. This should have sizes defined or else FLOT can't do the drawing.
 * @param xLabels: the labels for the x - axis
 * @param yLabels: the labels for the y - axis
 * @param seriesArray: the actual data to be used by FLOT
 * @param data_info: additional information about each node. Used when hovering over a node
 * @param min_color: minimum color, used for gradient
 * @param max_color: maximum color, used for gradient
 * @param backPage: page where visualizers fired from overlay should take you back.
 */
function _updatePlotPSE(canvasId, xLabels, yLabels, seriesArray, data_info, min_color, max_color, backPage, d3Data_info) {

    // why is it that we don't associate the labels into the attributes of the actual data_info or seriesArray?
    // where does the seriesArray structure get created?
    _PSE_minColor = min_color;
    _PSE_maxColor = max_color;
    PSE_nodesInfo = data_info;
    PSE_d3NodesInfo = d3Data_info;
    _PSE_plotOptions = {
        margins: { // is this the correct way to be doing margins? It's just how I have in the past,
            top: 20,
            bottom: 40,
            left: 20,
            right: 50
        },
        xaxis: {
            labels: xLabels
        },
        yaxis: {
            labels: yLabels
        }

    };
    _d3PSE_plot = d3Plot("#" + canvasId, seriesArray, $.extend(true, {}, _PSE_plotOptions), backPage);
}


function d3Plot(placeHolder, data, options, pageParam) {

    /*************************************************************************************************************************
     *   what follows here first are functions meant to assist in the plotting of results correctly with d3 as our plotting language.
     *************************************************************************************************************************/


    //these lines are cleanup for artifacts of the conversion that aren't behaving nicely, they should eventually be removed because they are just treating the symptoms.
    if (d3.select(".outerCanvas").empty() != true) {
        d3.selectAll(".outerCanvas").remove()
    }
    if (d3.selectAll("#main_div_pse")[0].length != 1) {
        var oneOrMoreDiv = d3.selectAll("div > div.flex-wrapper"); //index necessary because selection is an array with two elements, and second is unneccessary

        if (oneOrMoreDiv[0].length > 1) {
            oneOrMoreDiv[0][1].remove()
        } else {
            oneOrMoreDiv[0][0].remove()
        }
    }

    /*
     * The whole point of this function is to make the graph appear less crowded towards the axes, and each other dot. The point of discriminating between x and y is stylistic choice.
     */
    function createScale(xORy, labelArr) {
        if (xORy === "x") {
            var [lowerExtent,upperExtent] = d3.extent(labelArr),
                extentPadding = ((upperExtent - lowerExtent) * .10) / 2, // this multiplication factor controls how much the dots are gathered together
                [padLo,padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];


                var newScale = d3.scale.linear()
                    .domain([padLo, padUp])
                    .range([_PSE_plotOptions.margins.left, innerWidth - _PSE_plotOptions.margins.right]);
            }
        else {
            var [lowerExtent,upperExtent] = d3.extent(labelArr),
                extentPadding = ((upperExtent - lowerExtent) * .35) / 2,
                [padLo,padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];


                var newScale = d3.scale.linear()
                    .domain([padLo, padUp])
                    .range([innerHeight - (_PSE_plotOptions.margins.bottom), _PSE_plotOptions.margins.top]);

            }
            return newScale
        }


    /*
     * this makes a large range in the form of an array of values that configure to the proper step value that the ticks would otherwise be spaced at.
     * So far this is the only method that I've figured out to create grid lines for the graph background that can work for the extensive panning the user might be interested in.
     */

    function createRange(arr, step) {
        if (step == undefined) {
            step = arr[1] - arr[0];
        }
        return d3.range(-50 + arr[1], 50 + arr[1], step)
    }

    /*
     * This function generates an x or y axis structure with floating point accuracy up to 2 decimals. This will be used in slightly varying ways for the grid lines, and
     * the typical axes of the graph.
     */
    function createAxis(xORy, step) {
        if (xORy === "x") { // should I be creating the whole axis inside here, or should I simply return the axis that has the parts to be customized and called later
            newAxis = d3.svg.axis().scale(xScale)
                .orient("bottom")
                .tickValues(createRange(_PSE_plotOptions.xaxis.labels, step))
                .tickFormat(d3.format(",.2f"));
            return newAxis
        }
        else {
            newAxis = d3.svg.axis().scale(yScale)
                .orient("left")
                .tickValues(createRange(_PSE_plotOptions.yaxis.labels, step))
                .tickFormat(d3.format(",.2f")); // this means add in , when thousands are used, and look for 2 digits past the decimal, with value considered float type
            return newAxis
        }
    }


    /*
     * This function uses an ajax call to retrieve the stored configurations for the filter tool. it simply removes the options that were already present in the select bar
     * and loads in an updated version of them. The whole point of the for loop is to load in the options for multiple select bars if they are present in greater numbers than 1.
     */
    function getFilterSelections() {
        doAjaxCall({
            type: 'POST',
            url: '/flow/get_pse_filters',
            success: function (r) {
                for (var i = 0; i < d3.selectAll(".action-store")[0].length; i++) { // note the indexing due to the selectAll returning a one ele array of multiple arrays
                    var selectElement = d3.select("#filterSelect" + i);
                    selectElement.selectAll("option").remove();
                    selectElement.html(r);// this is the best way that i could come up with to separate out the returned elements
                }

            },
            error: function () {
                displayMessage("couldn't load the selection bar", "errorMessage")
            }
        })
    }

    /*
     * This function is more or less the same as the one above it except there is no need to account for multiple select bars.
     */
    function getContourSelections() {
        doAjaxCall({
            type: 'POST',
            url: '/flow/get_pse_filters',
            success: function (r) {
                var selectElement = d3.select("#contourSelect");
                selectElement.selectAll("option").remove();
                selectElement.html(r);// this is the best way that i could come up with to separate out the returned elements


            },
            error: function () {
                displayMessage("couldn't load the selection bar", "errorMessage")
            }
        })
    }

    /*
     * This function is what helps coordinate the new positions of each of our result dots when the user pans the graph, or zooms in and out.
     * the factor section prevents the dots from overlapping each other after extreme amounts of zooming, but it makes it possible to scale up
     * small sections of a great big batch result.
     */
    function moveDots() {
        circles
            .transition()
            .attr({
                r: function (d) {
                    var factor = xzoom.scale() * yzoom.scale()
                    if (factor > 2.5) {
                        return d.points.radius * 2.5;
                    } else if (factor < .5) {
                        return d.points.radius * .5
                    } else {
                        return d.points.radius * factor
                    }


                },
                cx: function (d) {
                    return xScale(d.coords.x)
                },
                cy: function (d) {
                    return yScale(d.coords.y)
                }



            })
    }


    /*
     * This is primarily a filter tool function. It is responsible for attaching the "on("change")" function to new select bars, saving behavior,
     * and making sure that they are up to date with the configuration options that the users have created.
     * Worth noting that there is an incremental id to keep track of which of the select bars actions are being executed on.
     * lastly, this also introduces the expected behavior for the "remove options" button that comes with each filter bar group after the first.
     */
    function refreshOnChange() {

        d3.selectAll(".filterSelectBar").on("change", function () { // why wont this execute on all selection bars?

            var filterSpecs = d3.select(this).property("value").split(','),
                filterType = filterSpecs[1].slice(0, -1), // error was coming up due to the extra number tagged onto the end of the button id, this keeps just the part that matters,could break if more than 10 filters rows are created
                filterNot = filterSpecs[2],
                incrementId = d3.select(this).property("id").slice(-1); //threshold value (type float) is stored first index, and then the type (string)
            d3.select("input#threshold" + incrementId).property("value", parseFloat(filterSpecs[0]));
            d3.select("input[type='radio']#" + filterType + incrementId).property("checked", true);
            d3.select("#Not" + incrementId).property('checked', filterNot);
        });

        d3.selectAll(".action-store").on("click", function () { // this might not be currently specific enough to save only the filter configs, I don't want this to fire when I click on a contour save button.
            var incrementId = d3.select(this).property("id").slice(-1),
                usrSelectedName = d3.select('#overlayNameInput' + incrementId).property('value'),
                incoming_values = {
                    threshold_value: d3.select('input#threshold' + incrementId).property('value'),
                    threshold_type: d3.select('input[name="thresholdType"]:checked').property('id'),
                    not_presence: d3.select("#Not" + incrementId).node().checked
                };
            doAjaxCall({
                type: 'POST',
                url: '/flow/store_pse_filter/' + usrSelectedName,
                data: incoming_values,
                success: function (r) {
                    getFilterSelections();
                    d3.select('#overlayNameInput' + incrementId).property('value', '')
                },
                error: function () {
                    displayMessage('could not store the selected text', 'errorMessage')
                }

            })
        });

        d3.selectAll(".action-minus").on("click", function () {
            d3.select(this).node().parentNode.remove()
        })
    }

    /*
     * this function is for the contour tool again. It simply takes the stored values of the selected option, and recreates the configuration in the elements of the dropdown
     * (filling in rate of change input, checking the not button if necessary, selecting the correct contour type from color&size).
     */
    function enactSelection() {
        d3.select("#contourSelect").on("change", function () {
            var filterSpecs = d3.select(this).property('value').split(","),
                filterType = filterSpecs[1],
                filterValue = filterSpecs[0],
                filterNot = filterSpecs[2];
            d3.select('input[name="RateOfChangeType"]#' + filterType).property("checked", true);
            d3.select('input#rateOfChangeInput').property("value", filterValue)
            d3.select("#notButton").property('checked', filterNot);
        })
    }


    /*
     * simply but necessary function to help us keep track of the results for things like transparentDots below. Without a key value for each result, specific entries failing to pass filter
     * wouldn't be accurately modified.
     */
    function getKey(d) {
        return d.key
    }

    /*
     * create a selection from the data that has been removed for not passing the filter criteria, and then transition it to having a high level of transparency.
     */
    function transparentDots() {
        d3.selectAll("circle").data(workingData, getKey).exit()
            .transition()
            .duration(500)
            .attr("fill-opacity", ".1")
    }


    /*
     * This function tells our axis, grid, and dots to react to the panning events that the users creates when they drag the clicked cursor across either of the axes. The call
     * simply re evaluates what should be displayed given the change in the graph.
     */
    function xzoomed() {
        d3.select("#xAxis").call(xAxis);
        d3.select("#xGrid").call(xGrid);
        moveDots()
    }

    function yzoomed() {
        d3.select("#yAxis").call(yAxis);
        d3.select("#yGrid").call(yGrid);
        moveDots()
    }

    /*
     * This function is only taking arguments from brushed space of graph, and the step slider bars of the explore drop down. With these pieces of information it draws horizontal
     * and vertical lines in the correct positions to fill in the Brushed area of the graph. The point is to provide illustration to the user of where results will be placed if
     * an actual exploration were launched.
     */
    function lineFillBrush(span, steps) {
        lineFunc = d3.svg.line()
            .x(function (d) {
                return d.x
            })
            .y(function (d) {
                return d.y
            })
            .interpolate("linear");

        canvas.append("g")
            .attr("id", "brushLines")
        var lineData;

        for (var xVal of d3.range(span[0][0], span[1][0], steps[0])) {
            lineData = [{x: xScale(xVal), y: yScale(span[0][1])}, {x: xScale(xVal), y: yScale(span[1][1])}];
            d3.select("#brushLines").append("path")
                .attr("d", lineFunc(lineData))
                .attr("stroke", "blue")
                .attr("stroke-width", ".5px")
                .attr("fill-opacity", ".1")
                .attr("fill", "none")
                .attr("id", "brushLine");
        }
        for (var yVal of d3.range(span[0][1], span[1][1], steps[1])) {
            lineData = [{x: xScale(span[0][0]), y: yScale(yVal)}, {x: xScale(span[1][0]), y: yScale(yVal)}];
            d3.select("#brushLines").append("path")
                .attr("d", lineFunc(lineData))
                .attr("stroke", "blue")
                .attr("stroke-width", ".5px")
                .attr("fill-opacity", ".1")
                .attr("fill", "none")
                .attr("id", "brushLine");
        }


    }

    /*
     * This function uses the ColSch gradient function to determine the fill color for each dot in the graph.
     */
    function returnfill(weight) {

        var colTest = ColSch_getGradientColorString(weight, _PSE_minColor, _PSE_maxColor).replace("a", ""), // the a creates an error in the color scale creation, so it must be removed.
            d3color = d3.rgb(colTest); // turn color string into a d3 compatible form.
            return d3color

    }

    /*
     * This function generates range labels to assist the user in inputting a threshold for the contour tool. It simply goes through the neighbor options for every dot in the graph
     * and determines what the differences are in size and color metric. Then the function simply returns an object of the max&mins of color and size comparisons for the labels.
     */
    function calcDiff() {
        var [maxDiffCol,minDiffCol,minDiffSize,maxDiffSize,] = [0, Infinity, Infinity, 0] //initializing the values to be returned
        allNeighbors = compareToNeighbors(structure, steps, inclusiveX, inclusiveY, {
            type: 'Color',
            value: 0,
            not: false
        }, PSE_d3NodesInfo);//the type here isn't important actually
        for (ob of allNeighbors) {
            for (neighborStr of ob.neighbors) {
                var [xNeighborCoord,yNeighborCoord]  = neighborStr.split(" "),
                    neighborOb = workingData.filter(function (dataOb) {
                        return dataOb.coords.x == xNeighborCoord && dataOb.coords.y == yNeighborCoord
                    }),
                    neighborSize = neighborOb[0].points.radius,
                    neighborColorWeight = PSE_d3NodesInfo[xNeighborCoord][yNeighborCoord].color_weight,
                    obColorWeight = PSE_d3NodesInfo[ob.focalPoint.coords.x][ob.focalPoint.coords.y].color_weight,
                    colorDiff = Math.abs(obColorWeight - neighborColorWeight),
                    min_size = +d3.select("#minShapeLabel").node().innerHTML,
                    max_size = +d3.select("#maxShapeLabel").node().innerHTML,
                    sizeScale = d3.scale.linear() //todo shift sizescale into the scope of the head of the d3plot function variables, it's used in a lot of places.
                        .range([min_size, max_size]) // these plus signs convert the string to number
                        .domain(d3.extent(workingData, function (d) {
                            return +d.points.radius
                        })),
                    sizeDiff = Math.abs(sizeScale(+neighborSize) - sizeScale(+ob.focalPoint.points.radius)); // gosh I need to have the radius to size converter on hand nearby all the time...
                if (maxDiffCol < colorDiff) {
                    maxDiffCol = colorDiff
                }
                if (minDiffCol > colorDiff && colorDiff != 0) {
                    minDiffCol = colorDiff

                }
                if (maxDiffSize < sizeDiff) {
                    maxDiffSize = sizeDiff

                }
                if (minDiffSize > sizeDiff && sizeDiff != 0) {
                    minDiffSize = sizeDiff
                }

            }
        }
        return {size: [minDiffSize, maxDiffSize], color: [minDiffCol, maxDiffCol]}
    }


    /*******************************************************************************
     *  this section below is devoted to variable declaration, and graph assembly.
     ******************************************************************************/

    var myBase, workingData, canvasDimensions, canvas, xScale, yScale, xAxis, yAxis, circles, brush,
        dotsCanvas, innerHeight, innerWidth, toolTipDiv, zoom, datatypeGID, structure, inclusiveX, inclusiveY, steps;
    myBase = d3.select(placeHolder);
    workingData = sortResults(data); //structure expects the sorting that this function performs.
    [inclusiveX, inclusiveY] = constructLabels(workingData);
    steps = {
        x: [+(inclusiveX[1] - inclusiveX[0]).toFixed(4)] // prevent any wierd float point arithmetic
        , y: [+(inclusiveY[1] - inclusiveY[0]).toFixed(4)]
    };
    structure = createStructure(workingData, inclusiveX, inclusiveY);
    for (ind in workingData) {
        workingData[ind].key = parseFloat(ind)
    }
    ;
    canvasDimensions = {h: parseInt(myBase.style("height")), w: parseInt(myBase.style("width"))};
    innerHeight = canvasDimensions.h - _PSE_plotOptions.margins.top;
    innerWidth = canvasDimensions.w - _PSE_plotOptions.margins.left;
    datatypeGID = d3.select("#datatype-group-gid").property("value");
    xScale = createScale("x", inclusiveX);
    yScale = createScale("y", inclusiveY);
    yzoom = d3.behavior.zoom() // these zooms are to create the panning behavior that the user employs for zooming in/out or side to side adjustments of graph.
        .y(yScale)
        .on("zoom", yzoomed);
    xzoom = d3.behavior.zoom()
        .x(xScale)
        .on("zoom", xzoomed);
    canvas = myBase.append("svg") // now we establish the actual selection to generate the background for our graph, using dimensions taken from above.
        .attr({
            class: "outerCanvas",
            height: canvasDimensions.h,
            width: canvasDimensions.w
        })
        .append("g")
        .attr("transform", "translate( " + _PSE_plotOptions.margins.left + "," + _PSE_plotOptions.margins.top + " )"); // means that everything that is child of this selection will be adjusted by margins
    canvasClip = canvas.append("svg:clipPath") //thes objects limit what is currently displayed on the graph and the axes. Keep in mind that without these the axes and grids would stretch beyond where we expect to see them.
        .attr("id", "genClip")
        .append("svg:rect")
        .attr("id", "clipRect")
        .attr("x", _PSE_plotOptions.margins.left)
        .attr("y", _PSE_plotOptions.margins.top)
        .attr("width", innerWidth - _PSE_plotOptions.margins.left - _PSE_plotOptions.margins.right)
        .attr("height", innerHeight - _PSE_plotOptions.margins.bottom - _PSE_plotOptions.margins.top);
    xAxisClip = canvas.append("svg:clipPath")
        .attr("id", "xClip")
        .append("svg:rect")
        .attr("x", _PSE_plotOptions.margins.left)
        .attr("y", 0)
        .attr("width", innerWidth - _PSE_plotOptions.margins.left - _PSE_plotOptions.margins.right)
        .attr("height", _PSE_plotOptions.margins.bottom);
    yAxisClip = canvas.append("svg:clipPath")
        .attr("id", "yClip")
        .append("svg:rect")
        .attr("x", -_PSE_plotOptions.margins.left * 2)// these two areas are simply selected for what they accomplish visually. I wonder if there could be a real connection to the values used for arranging the canvas
        .attr("y", _PSE_plotOptions.margins.top)
        .attr("width", _PSE_plotOptions.margins.right)//
        .attr("height", innerHeight - _PSE_plotOptions.margins.bottom - _PSE_plotOptions.margins.top);

    toolTipDiv = d3.select(".tooltip"); // this is the hover over dot tool tip which displays information stored in the PSE_nodeInfo variable.
    xAxis = createAxis("x", undefined); // these lines create the axes and grids
    xGrid = createAxis("x", undefined)
        .tickSize(innerHeight, 0, 0)
        .tickFormat(""); //means don't show any values at the base line of the axis
    yAxis = createAxis("y", undefined);
    yGrid = createAxis("y", undefined)
        .tickSize(-innerWidth, 0, 0)
        .tickFormat("");


    canvas.append("g")
        .attr("id", "yGrid")
        .attr("clip-path", "url(#genClip)") // applying the clip to limit extent of displayed grid
        .attr("transform", "translate (0,0)")
        .style("stroke", "gray")
        .style("stroke-opacity", ".5")
        .call(yGrid);

    canvas.append("g")
        .attr("id", "xGrid")
        .attr("clip-path", "url(#genClip)")
        .attr("transform", "translate (0,0)")
        .style("stroke", "gray")
        .style("stroke-opacity", ".5")
        .call(xGrid);

    canvas.append("g") // the tricky part here is to apply the clip where the xaxis was before the transform... more g tag strangeness
        .attr("id", "xAxis")
        .attr("clip-path", "url(#xClip)")
        .attr("transform", "translate (0," + ( innerHeight - _PSE_plotOptions.margins.bottom ) + ")")
        .call(xAxis)
        .call(xzoom); // tie the panning behaviour to this element
    canvas.append("g")
        .attr("id", "yAxis")
        .attr("clip-path", "url(#yClip)")
        .attr("transform", "translate (" + _PSE_plotOptions.margins.left + " ,0)")
        .call(yAxis)
        .call(yzoom);

    dotsCanvas = canvas.append("svg") // generate the SVG container for the plotted dots.
        .classed("dotsCanvas", true)
        .attr({
            height: innerHeight,
            width: innerWidth
        })
        .attr("clip-path", "url(#genClip)");
    circles = dotsCanvas.selectAll("circle").data(workingData, getKey).enter().append("circle") // bind the data using d3, and use data to assign specific attributes to each dot
        .attr({
            r: function (d) {
                return d.points.radius
            },
            cx: function (d) {
                return xScale(d.coords.x) // the xScale converts from data values to pixel values for appropriate actual placement in the graphed space.
            },
            cy: function (d) {
                return yScale(d.coords.y)
            },
            fill: function (d) {
                var nodeInfo = PSE_d3NodesInfo[d.coords.x][d.coords.y];
                if (nodeInfo.tooltip.search("PENDING") == -1 && nodeInfo.tooltip.search("CANCELED") == -1) { // this prevents code from trying to find reasonable color values when simulation results haven't been generated for them
                    color = returnfill(nodeInfo.color_weight);
                }
                else {
                    var color = d3.rgb("black"); // leave pending results blacked out if not complete.
                }
                return color // otherwise fill out with color in keeping with scheme.
            }

        });


    /*************************************************************************************************************************
     *   what follows here is the associated code for mouse events, and user generated events
     *************************************************************************************************************************/

    /*
     * This indicates what should be done at the clicking of the contour button.
     */
    d3.select('#Contour').on("click", function () {
        var contourDiv = d3.select("#contourDiv");
        if (contourDiv.style("display") == "none") {
            contourDiv.style("display", "block"); // makes the contour dropdown menu visible
            getContourSelections()
            enactSelection();
            var tipFillin = calcDiff(); // this is the label filling in for threshold of rate of change
            d3.select("label[for='Size']").html('Size --> (' + tipFillin.size[0].toExponential(4) + ", " + tipFillin.size[1].toExponential(4) + ")")
            d3.select("label[for='Color']").html('Color --> (' + tipFillin.color[0].toExponential(4) + ", " + tipFillin.color[1].toExponential(4) + ")")
        }


        else {
            contourDiv.style("display", "none") // removes the contour menu from display
        }

    })


    /*
     * executes the drawing of lines where the rate of change passes the users criteria.
     */
    d3.select("#contourGo").on("click", function () {

        function drawCompLines(relationOb) {

            var lineFunc = d3.svg.line()
                .x(function (d) {
                    return d.x
                })
                .y(function (d) {
                    return d.y
                })
                .interpolate("linear");

            var lineCol = d3.rgb(Math.random() * 255, Math.random() * 255, Math.random() * 255); // this will give us a way to keep contours on the page between comparisons
            for (var currentOb of relationOb) {
                for (var neighbor of currentOb.neighbors) {
                    var neighborsCoords = neighbor.split(" ").map(function (ele) {
                            return +ele
                        }), //breakdown of line: separate and convert coordinates from string to float before assigning to variables.
                        xNeighbor = xScale(neighborsCoords[0]),
                        yNeighbor = yScale(neighborsCoords[1]),
                        obX = xScale(currentOb.focalPoint.coords.x),
                        obY = yScale(currentOb.focalPoint.coords.y),
                        deltaX = (xNeighbor - obX), //simple final minus initial for change
                        deltaY = (yNeighbor - obY),
                        midPoint = {x: obX + deltaX / 2, y: obY + deltaY / 2},
                        startCoord = {x: midPoint.x + deltaY / 2, y: midPoint.y - deltaX / 2}, // gives instruction as to what direction and how far from the midpoint to establish start&end of line
                        endCoord = {x: midPoint.x - deltaY / 2, y: midPoint.y + deltaX / 2};
                    d3.select(".dotsCanvas").append("path")
                        .attr("d", lineFunc([startCoord, endCoord])) // generate the svg path for the line
                        .attr("stroke", lineCol)
                        .attr("stroke-width", ".5px")
                        .attr("fill-opacity", ".1")
                        .attr("fill", "none")
                        .attr("id", "contourLine");
                }
            }

        }

        var criteria = { //retrieve the user specifications for the contour run
            type: d3.select('input[name="RateOfChangeType"]:checked').node().id,
            value: +d3.select('input#rateOfChangeInput').node().value,
            not: d3.select("#notButton").property('checked')
        }

        neighborsObjct = compareToNeighbors(structure, steps, inclusiveX, inclusiveY, criteria, PSE_d3NodesInfo) // use function from the alternateDatastructure.js to determine which dots and neigbors will be separated by lines.
        drawCompLines(neighborsObjct)
    })

    d3.select("#contourClear").on("click", function () {
        d3.selectAll('#contourLine').remove()
    });

    /*
     * store the user specified configuration for a contour run as session attribute
     */
    d3.select("#saveContourConfig").on("click", function () {

        var usrSelectedName = d3.select('#contourNameInput').property('value'),
            incoming_values = {
                threshold_value: d3.select('input#rateOfChangeInput').node().value,
                threshold_type: d3.select('input[name="RateOfChangeType"]:checked').node().id,
                not_presence: d3.select("#notButton").node().checked
            };
        doAjaxCall({
            type: 'POST',
            url: '/flow/store_pse_filter/' + usrSelectedName,
            data: incoming_values,
            success: function (r) {
                getFilterSelections();
                d3.select('#contourNameInput').property('value', '')
            },
            error: function () {
                displayMessage('could not store the selected text', 'errorMessage')
            }

        })
    });

    /*
     * specifics for drawing the explore menu upon clicking the explore button, and the generation of brush for the graph.
     */
    d3.select("#Explore").on("click", function () {
        var exploreDiv = d3.select("#exploreDiv");
        if (exploreDiv.style("display") == "none") {
            exploreDiv.style("display", "block");
        }


        else {
            exploreDiv.style("display", "none")
        }


        /*
         * this function is what is executed upon the release of the mouse when selecting a region of the graph. Here we update the range inputs for the extent of the brushed
         * area. Furthermore we extend the behavior of the sliders to dynamically change the spacing of the grid lines within the brushed area of the graph.
         */
        function expBrushStop() {
            d3.select("#brushLines").remove();

            var extent = exploreBrush.extent();
            var xRange = Math.abs(extent[0][0] - extent[1][0]),
                yRange = Math.abs(extent[0][1] - extent[1][1]),
                xSteps = d3.select("input[name='xStepInput']").node().value,
                ySteps = d3.select("input[name='yStepInput']").node().value;
            d3.select('#lowX').node().value = extent[0][0].toFixed(4);
            d3.select('#upperX').node().value = extent[1][0].toFixed(4);
            d3.select('#lowY').node().value = extent[0][1].toFixed(4);
            d3.select('#upperY').node().value = extent[1][1].toFixed(4);

            lineFillBrush(extent, [xSteps, ySteps]) // draw initial grid lines when mouse button is released.
            var elemSliderA = $('#XStepSlider'); // this is included here to make the sliders affect the drawing of the grid lines dynamically
            elemSliderA.slider({
                min: 0, max: extent[1][0] - extent[0][0], step: .0001, value: xSteps,
                slide: function (event, ui) {
                    xSteps = ui.value;
                    d3.select('input[name="xStepInput"]').property('value', ui.value);
                    d3.select("#brushLines").remove() // remove the previously drawn lines to prevent confusion.
                    lineFillBrush(extent, [xSteps, ySteps]) // redraw grid lines

                }
            });
            var elemSliderB = $('#YStepSlider');
            elemSliderB.slider({
                min: 0, max: extent[1][1] - extent[0][1], step: .0001, value: ySteps,
                slide: function (event, ui) {
                    ySteps = ui.value;
                    d3.select('input[name="yStepInput"]').property('value', ui.value);
                    d3.select("#brushLines").remove()
                    lineFillBrush(extent, [xSteps, ySteps])
                }
            })
        }


        var exploreBrush = d3.svg.brush() // generate the brush
            .x(xScale)
            .y(yScale)
            .on("brushend", expBrushStop);
        if (d3.select(".brush").empty() == true) { // if there not already a brush attach it to the canvas
            canvas.append("g")
                .attr("class", "brush")
                .call(exploreBrush)
                .selectAll("rect");
            d3.select('.extent').style({ // create temporary stylish rules for the brush.
                stroke: '#4dbbbb',
                'fill-opacity': '.125',
                'shape-rendering': 'crispEdges'
            });

        } else { // if we found a brush, then we need to remove it because if we get here we are closing the menu
            d3.select(".brush").remove();
            d3.select("#brushLines").remove();
        }


    });


    /*
     * when we click the "Explore Section" button send the info from the drop down menu to be stored as a session attribute.
     */
    d3.select("#exploreGo").on("click", function () {
        var xRange = [d3.select("#lowX").node().value, d3.select("#upperX").node().value],
            yRange = [d3.select("#lowY").node().value, d3.select("#upperY").node().value],
            xStep = d3.select("input[name='xStepInput']").node().value,
            yStep = d3.select("input[name='yStepInput']").node().value;

        doAjaxCall({
            type: "POST",
            url: "/flow/store_exploration_section/" + [xRange, yRange] + "/" + [xStep, yStep] + "/" + datatypeGID,
            success: function () {
                displayMessage("successfully stored exploration details")
            },
            error: function () {
                displayMessage(error, "couldn't store the exploration details")
            }

        })
    });

    /*
     * display the filter drop down on clicking the filter button in the tool bar. Populate the selections, and selection behavior if there are actually entries.
     */
    d3.select("#Filter").on("click", function () {


        var filterDiv = d3.select("#FilterDiv"),
            idNum = d3.selectAll("#threshold").length;
        if (filterDiv.style("display") == "none") {
            filterDiv.style("display", "block");
            getFilterSelections()
            refreshOnChange()
        }


        else {
            filterDiv.style("display", "none")
        }
    });

    /*
     * This is what happens when we click the "start filter" button. First the user criteria bars are concatenated into a string for boolean comparison through eval. Then we
     * retrieve the actual thresholds for comparisons
     */
    d3.select("#filterGo").on("click", function () { // will the filtering also be any metric of option?

        function concatCriteria() {
            var filterFields = d3.selectAll(".toolbar-inline#filterTools > li")[0],
                concatStr = '';
            for (var set of filterFields) {
                var groupSelection = d3.select(set),
                    logicalPresence = groupSelection.select("input[name='logicButton']:checked").node(),
                    logicalOperator;
                if (logicalPresence != null) {
                    if (logicalPresence.id == 'Or') {
                        logicalOperator = ' ||'
                    } else {
                        logicalOperator = ' &&'
                    }
                } else {
                    logicalOperator = ''
                }
                var thresholdValue = groupSelection.select(".thresholdInput").property('value'),
                    notPreference = (groupSelection.select("input[name='notButton']").property('checked')) ? "<" : ">", //gives us true or false
                    filterType = groupSelection.select("input[name='thresholdType']:checked").property('id').slice(0, -1); // gives either 'Color' or 'Size'
                concatStr += logicalOperator + ' ' + filterType + ' ' + notPreference + ' ' + thresholdValue;

            }

            return concatStr

        }

        var min_size = +d3.select("#minShapeLabel").node().innerHTML,
            max_size = +d3.select("#maxShapeLabel").node().innerHTML,
            sizeScale = d3.scale.linear()
                .range([min_size, max_size]) // these plus signs convert the string to number
                .domain(d3.extent(workingData, function (d) {
                    return +d.points.radius
                }))


        for (var circle of d3.selectAll('circle')[0]) {
            var radius = sizeScale(+circle.attributes.r.value),
                filterString = concatCriteria(),
                data = circle.__data__,
                colorWeight = PSE_d3NodesInfo[data.coords.x][data.coords.y].color_weight,
                filterString = filterString.replace(/Size/g, radius).replace(/Color/g, colorWeight);
            if (!eval(filterString)) { // this phrasing is now persistent phrased, meaning that the data that the user wants to keep doesn't pass.
                var repInd = workingData.indexOf(data)
                if (repInd != -1) { // keeps the function from repeatedly detecting transparent dots, but removing others from workingData that pass the criteria
                    workingData.splice(repInd, 1) //this will remove the data from the group, and then it can be selected for below in the transparent dots
                }
            }
        }
        transparentDots()


    });


    /*
     * create a new row of filter criteria if the user selects "add options"
     */
    d3.select("#addFilterOps").on("click", function () {
        var nextRowId = d3.selectAll("button.action-store")[0].length;
        doAjaxCall({
            type: "POST",
            url: "/flow/create_row_of_specs/" + nextRowId + "/", //remember if you experience an error about there now being a row for one(), there is some silly typo sitting around, so go and check everything with the working examples.
            success: function (r) {
                var newLiEntry = d3.select("#FilterDiv > ul").append("li").html(r)
                getFilterSelections()
                refreshOnChange()
            },
            error: function () {
                displayMessage("couldn't add new row of filter options", "errorMessage")
            }
        })

    })


    /*
     * provides basis for behavior when the cursor passes over a result. we get shown the message that is stored as information in the nodeInfo.tooltip
     */
    d3.selectAll("circle").on("mouseover", function (d) {
        var nodeInfo = PSE_d3NodesInfo[d.coords.x][d.coords.y];
        var toolTipText = nodeInfo.tooltip.split("&amp;").join("&").split("&lt;").join("<").split("&gt;").join(">");
        toolTipDiv.html(toolTipText);
        toolTipDiv.style({
            position: "absolute",
            left: (d3.event.pageX) + "px",
            top: (d3.event.pageY - 100) + "px",
            display: "block",
            'background-color': '#C0C0C0',
            border: '1px solid #fdd',
            padding: '2px',
            opacity: 0.80
        })
    })
        .on("mouseout", function (d) {
            toolTipDiv.transition()
                .duration(300)
                .style("display", "none")
        });

    /*
     * this is the behavior for when a result is clicked. We have to bring up the window that gives the user more options relating to the specific result clicked upon.
     */
    d3.selectAll("circle").on("click", function (d) {
        var nodeInfo = PSE_d3NodesInfo[d.coords.x][d.coords.y];
        if (nodeInfo.dataType != undefined) {
            displayNodeDetails(nodeInfo['Gid'], nodeInfo['dataType'], pageParam); // curious because backPage isn't in the scope, but appears to work.
        }
    });


    /*
     * This is the function that specifies how to convert the graph into an exportable file.
     */
    d3.select("#ctrl-action-export").on("click", function (d) {

    })


}
/*
 * Do a redraw of the plot.
 */
function redrawPlot(plotCanvasId) {
    if (backPage == null || backPage == '') {
        var backPage = get_URL_param('back_page');
    }
    PSE_mainDraw('main_div_pse', backPage)

}


/*
 * create a colored legend for the current colorScheme and data results, and place it in the upper right of the graphed space.
 */
function updateLegend(minColor, maxColor) {
    var legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend");
    legendHeight = d3.select("#table-colorWeightsLegend").node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight);
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, legendHeight);
    d3.selectAll(".matrix-legend")
        .style({
            position: 'absolute',
            top: '118px',
            right: '25px'
        })
    d3.select("#colorWeightsLegend")
        .style("right", "85px")
}


function PSEDiscreteInitialize(labelsXJson, labelsYJson, series_array, dataJson, backPage, hasStartedOperations,
                               min_color, max_color, min_size, max_size, d3DataJson) {


    var labels_x = $.parseJSON(labelsXJson);
    var labels_y = $.parseJSON(labelsYJson);
    var data = $.parseJSON(dataJson);
    series_array = typeof (series_array) == "string" ? $.parseJSON(series_array) : series_array;
    var d3Data = typeof (d3DataJson) == "string" ? $.parseJSON(d3DataJson) : d3DataJson;

    min_color = parseFloat(min_color); // todo run a batch of simulations part of the way,  and then cancel to see what the result looks like.
    max_color = parseFloat(max_color);
    min_size = parseFloat(min_size);
    max_size = parseFloat(max_size);

    ColSch_initColorSchemeGUI(min_color, max_color, function () { //this now doesn't create error in simulator panel, why?
        _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage, d3Data);
    });

    function _fmt_lbl(sel, v) {
        $(sel).html(Number.isNaN(v) ? 'not available' : toSignificantDigits(v, 3));
    }

    _fmt_lbl('#minColorLabel', min_color);
    _fmt_lbl('#maxColorLabel', max_color);
    _fmt_lbl('#minShapeLabel', min_size);
    _fmt_lbl('#maxShapeLabel', max_size);

    if (Number.isNaN(min_color)) {
        min_color = 0;
        max_color = 1;
    }
    _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage, d3Data);
    updateLegend(min_color, max_color)


    if (hasStartedOperations) {
        setTimeout("PSE_mainDraw('main_div_pse','" + backPage + "')", 3000);
    }
}


/*
 * Take currently selected metrics and refresh the plot.
 */
function PSE_mainDraw(parametersCanvasId, backPage, groupGID) {

    if (groupGID == null) {
        // We didn't get parameter, so try to get group id from page
        groupGID = document.getElementById("datatype-group-gid").value;
    }
    if (backPage == null || backPage == '') {
        backPage = get_URL_param('back_page');
    }

    var url = '/burst/explore/draw_discrete_exploration/' + groupGID + '/' + backPage;
    var selectedColorMetric = $('#color_metric_select').val();
    var selectedSizeMetric = $('#size_metric_select').val();

    if (selectedColorMetric != '' && selectedColorMetric != null) {
        url += '/' + selectedColorMetric;
        if (selectedSizeMetric != '' && selectedSizeMetric != null) {
            url += '/' + selectedSizeMetric;
        }
    }


    doAjaxCall({
        type: "POST",
        url: url,
        success: function (r) {
            $('#' + parametersCanvasId).html(r);
        },
        error: function () {
            displayMessage("Could not refresh with the new metrics.", "errorMessage");
        }
    });
}


/*************************************************************************************************************************
 *            ISOCLINE PSE BELLOW
 *************************************************************************************************************************/


var serverURL = null;
var figuresDict = null;
var currentFigure = null;


/*
 * Do the actual resize on currentFigure global var, and a given width and height.
 */
function resizePlot(width, height) {
    if (currentFigure != null) {
        MPLH5_resize = currentFigure;
        do_resize(currentFigure, width, height);
        MPLH5_resize = -1;
    }
}

/*
 * Store all needed data as js variables so we can use later on.
 */
function initISOData(metric, figDict, servURL) {
    figuresDict = $.parseJSON(figDict);
    serverURL = servURL;
    currentFigure = figuresDict[metric];
    connect_manager(serverURL, figuresDict[metric]);
    $('#' + metric).show();
    initMPLH5CanvasForExportAsImage(figuresDict[metric]);
}

/*
 * On plot change update metric and do any required changes like resize on new selected plot.
 */
function updateMetric(selectComponent) {
    var newMetric = $(selectComponent).find(':selected').val();
    showMetric(newMetric);
    var pseElem = $('#section-pse');
    var width = pseElem.width() - 60;
    var height = pseElem.height() - 90;
    waitOnConnection(currentFigure, 'resizePlot(' + width + ', ' + height + ')', 200, 50);
}

/*
 * Update html to show the new metric. Also connect to backend mplh5 for this new image.
 */
function showMetric(newMetric) {
    for (var key in figuresDict) {
        $('#' + key).hide()
            .find('canvas').each(function () {
            if (this.drawForImageExport) {            // remove redrawing method such that only current view is exported
                this.drawForImageExport = null;
            }
        });
    }
    currentFigure = figuresDict[newMetric];
    connect_manager(serverURL, figuresDict[newMetric]);
    $('#' + newMetric).show();
    initMPLH5CanvasForExportAsImage(figuresDict[newMetric]);
}

/*
 * This is the callback that will get evaluated by an onClick event on the canvas through the mplh5 backend.
 */
function clickedDatatype(datatypeGid) {
    displayNodeDetails(datatypeGid);
}

/*
 * Update info on mouse over. This event is passed as a callback from the isocline python adapter.
 */
function hoverPlot(id, x, y, val) {
    $('#cursor_info_' + id).html('x axis:' + x + ' y axis:' + y + ' value:' + val);
}


function Isocline_MainDraw(groupGID, divId, width, height) {
    width = Math.floor(width);
    height = Math.floor(height);
    $('#' + divId).html('');
    doAjaxCall({
        type: "POST",
        url: '/burst/explore/draw_isocline_exploration/' + groupGID + '/' + width + '/' + height,
        success: function (r) {
            $('#' + divId).html(r);
        },
        error: function () {
            displayMessage("Could not refresh with the new metrics.", "errorMessage");
        }
    });
}


