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
//todo collapse plot to single svg with cutting path support
//todo investigate new series array structure that will make adding more dots easier
//todo create an exporting function that can save the figure
//todo ask about how to store overlays inside viewer associated with original file, and find way to load them when user specifies.

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
function _updatePlotPSE(canvasId, xLabels, yLabels, seriesArray, data_info, min_color, max_color, backPage) {

    _PSE_minColor = min_color;
    _PSE_maxColor = max_color;
    PSE_nodesInfo = data_info;
    _PSE_plotOptions = {
        series: {
            lines: {
                show: false
            },
            points: {
                lineWidth: 0,
                show: true,
                fill: true
            }
        },
        margins: { // is this the correct way to be doing margins? It's just how I have in the past,
            top: 20,
            bottom: 40,
            left: 15,
            right: 50
        },
        xaxis: {
            labels: xLabels, // is there a better way to get access to these values inside my plotting?
            min: -1,
            max: xLabels.length,
            tickSize: 1,
            tickFormatter: function (val) {
                if (val < 0 || val >= xLabels.length) {
                    return "";
                }
                return xLabels[val];
            }
        },
        yaxis: {
            labels: yLabels,
            min: -1,
            max: yLabels.length,
            tickSize: 1,
            tickFormatter: function (val) {
                if (val < 0 || val >= yLabels.length || yLabels[val] === "_") {
                    return "";
                }
                return yLabels[val];
            }
        },
        grid: {
            clickable: true,
            hoverable: true
        }

    };
    _d3PSE_plot = d3Plot("#" + canvasId, seriesArray, $.extend(true, {}, _PSE_plotOptions), backPage);

    //this has been commented out below so that I can see what I have done on the canvas after the above function has ended
    /*_PSE_plot = $.plot($("#" + canvasId), $.parseJSON(seriesArray), $.extend(true, {}, _PSE_plotOptions));
     changeColors(); // this will need to eventually have the addition of the d3 plot function
     $(".tickLabel").each(function () {
     $(this).css("color", "#000000");
     });
     //if you want to catch the right mouse click you have to change the flot sources
     // because it allows you to catch only "plotclick" and "plothover"
     applyClickEvent(canvasId, backPage);
     applyHoverEvent(canvasId);*/
}


function d3Plot(placeHolder, data, options, pageParam) {
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
    function createScale(xORy) {
        // should I incorporate some sort of testing for values before actually getting into the function?
        //todo relate the scaling factor to the radius values available
        if (xORy === "x") {
            var [lowerExtent,upperExtent] = d3.extent(_PSE_plotOptions.xaxis.labels),
                extentPadding = ((upperExtent - lowerExtent) * .10) / 2, // this multiplication factor controls how much the dots are gathered together
                [padLo,padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];

            if (padLo < 0) {
                var newScale = d3.scale.linear()
                    .domain([0, padUp])
                    .range([0, innerWidth - options.margins.right]);

            } else {
                var newScale = d3.scale.linear()
                    .domain([padLo, padUp])
                    .range([options.margins.left, innerWidth - options.margins.right]);
            }
            return newScale
        } else {
            var [lowerExtent,upperExtent] = d3.extent(_PSE_plotOptions.yaxis.labels),
                extentPadding = ((upperExtent - lowerExtent) * .35) / 2,
                [padLo,padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];

            if (padLo < 0) {
                var newScale = d3.scale.linear()
                    .domain([0, padUp])
                    .range([innerHeight - (options.margins.bottom), options.margins.top]);

            } else {
                var newScale = d3.scale.linear()
                    .domain([padLo, padUp])
                    .range([innerHeight - (options.margins.bottom), options.margins.top]);

            }
            return newScale
        }
    }

    function createAxis(xORy) {
        if (xORy === "x") { // should I be creating the whole axis inside here, or should I simply return the axis that has the parts to be customized and called later
            newAxis = d3.svg.axis().scale(xScale)
                .orient("bottom")
                .ticks(options.xaxis.max);
            return newAxis
        }
        else {
            newAxis = d3.svg.axis().scale(yScale)
                .orient("left")
                .ticks(options.yaxis.max);
            return newAxis
        }
    }


    function moveDots() {
        circles
            .transition()
            .attr({
                cx: function (d) {
                    return xScale(_PSE_plotOptions.xaxis.tickFormatter(d.data[0][0]))
                },
                cy: function (d) {
                    return yScale(_PSE_plotOptions.yaxis.tickFormatter(d.data[0][1]))
                }

                // return yScale(d.yCen) // why is this placing dots far below the bottom of the pane? Is the canvas dimension off?


            })
    }

    function workingDataRemove(index, dataObj) {
        for (i in dataObj) {
            if (dataObj[i].data[0] == index) {
                dataObj.splice(i, 1);
                return
            }
        }

    }


    function getKey(d) {
        return d.key
    }

    function transparentDots() {
        d3.selectAll("circle").data(workingData, getKey).exit()
            .transition()
            .duration(500)
            .attr("fill-opacity", ".1")
    }

    function zoomed() {
        d3.select("#xAxis").call(xAxis);
        d3.select("#yAxis").call(yAxis);
        moveDots()
    }

    function returnfill(weight) {
        var colTest = ColSch_getGradientColorString(weight, _PSE_minColor, _PSE_maxColor).replace("a", ""), // the a creates an error in the color scale creation
            d3color = d3.rgb(colTest);
        return d3color

    }


    var myBase, workingData, canvasDimensions, canvas, xScale, yScale, xRef, yRef, xAxis, yAxis, circles, brush,
        colScale, dotsCanvas, innerHeight, innerWidth, toolTipDiv, zoom;
    myBase = d3.select(placeHolder);
    workingData = $.parseJSON(data);
    for (ind in workingData) {
        workingData[ind].key = parseFloat(ind)
    }
    ;
    canvasDimensions = {h: parseInt(myBase.style("height")), w: parseInt(myBase.style("width"))};
    innerHeight = canvasDimensions.h - options.margins.top;
    innerWidth = canvasDimensions.w - options.margins.left;
    xScale = createScale("x");
    yScale = createScale("y");
    zoom = d3.behavior.zoom()
        .x(xScale)
        .y(yScale)
        .on("zoom", zoomed)
        .on("zoomend", moveDots);
    canvas = myBase.append("svg")
        .attr({
            class: "outerCanvas",
            height: canvasDimensions.h,
            width: canvasDimensions.w
        })
        .append("g")
        .attr("transform", "translate( " + options.margins.left + "," + options.margins.top + " )")
        .call(zoom);
    xRef = xScale.copy();
    yRef = yScale.copy();
    toolTipDiv = d3.select(".tooltip");
    xAxis = createAxis("x");
    yAxis = createAxis("y");
    // colScale = makeColScale();

    dotsCanvas = canvas.append("svg")
        .classed("dotsCanvas", true)
        .attr({
            height: innerHeight,
            width: innerWidth
        });

    circles = dotsCanvas.selectAll("circle").data(workingData, getKey).enter().append("circle") //todo make this a function so that it can be called after filter has removed dots to bring them back, and refresh workingdata
        .attr({
            r: function (d) {
                return d.points.radius
            },
            cx: function (d) {
                return xScale(_PSE_plotOptions.xaxis.tickFormatter(d.data[0][0]))
            },
            cy: function (d) {
                return yScale(_PSE_plotOptions.yaxis.tickFormatter(d.data[0][1]))
            },
            fill: function (d) {
                var nodeInfo = PSE_nodesInfo[d.data[0][0]][d.data[0][1]],
                    color = returnfill(nodeInfo.color_weight);
                return color
            }

        });


    canvas.append("g")
        .attr("id", "xAxis")
        .attr("transform", "translate (0," + ( innerHeight - _PSE_plotOptions.margins.bottom ) + ")")
        .call(xAxis);
    canvas.append("g")
        .attr("id", "yAxis")
        .attr("transform", "translate (" + _PSE_plotOptions.margins.left + " ,0)")
        .call(yAxis);
    // this is now the area that should allow for drawing the lines of the grid
    //todo again visual grid stuff. How should I go about making the grid fit the canvas better?
    // testing patch process


    d3.select("#Explore").on("click", function () { //todo deactivate the hand panning so that brush can be used
        function expBrushMove() {
            // var xRange
        }

        function expBrushStop() { // todo add sliders to the div that shows up
            if (exploreBrush.empty() == true) {
                explToolTip.style("display", "none")
            } else {
                var extent = exploreBrush.extent();
                var xRange = Math.abs(extent[0][0] - extent[1][0]),
                    yRange = Math.abs(extent[0][1] - extent[1][1]);
                explToolTip.style({
                    position: "absolute",
                    left: xScale(extent[1][0]) + _PSE_plotOptions.margins.left + "px", //this is the x cordinate of where the drag ended (assumption here is drags from left to right
                    top: yScale(extent[1][1]) + _PSE_plotOptions.margins.top + 100 + "px",
                    display: "block",
                    'background-color': '#C0C0C0',
                    border: '1px solid #fdd',
                    padding: '2px',
                    opacity: 0.80
                });
                d3.select("#xRange").text(xRange);
                d3.select("#yRange").text(yRange)
            }
        }

        var explToolTip = d3.select("#ExploreToolTip");

        var exploreBrush = d3.svg.brush()
            .x(xScale)
            .y(yScale)
            .on("brush", expBrushMove)
            .on("brushend", expBrushStop);
        if (d3.select(".brush").empty() == true) {
            canvas.append("g")
                .attr("class", "brush")
                .call(exploreBrush)
                .selectAll("rect");
        } else {
            d3.select(".brush").remove();
            explToolTip.style("display", "none"); // is this redundant with the above tooltip hider?
        }


    });

    d3.select("#Filter").on("click", function () { //todo standardize the id names for the div elements used for the various overlays.
        //todo ask lia if I'm going about adding the selector in the correct way.
        //todo ask lia how to debug the python aspects of this code. (breakpoints and introspection)
        //todo ask what the abc adapter is inside the flowcontroller
        //todo ask if the name parameter needs to be an element already in existence?
        //todo ask what is the treesession stored key?

        var filterDiv = d3.select("#FilterDiv"),
            idNum = d3.selectAll("#threshold").length;
        if (filterDiv.style("display") == "none") {
            filterDiv.style("display", "block")
            doAjaxCall({ // i believe that I know now that this would be the wrong ajax call just to make a selector also.
                type: 'POST',
                url: '/flow/testselectioncreator/testTextID/testButtonId',
                success: function (r) {
                    console.log(r)
                    // d3.select("#FilterDiv").append
                },
                error: function () {
                    displayMessage("couldn't load the selection bar", "errorMessage")
                }


            })
    }
else
    {
            filterDiv.style("display", "none")
        }
    });

    d3.select("#filterGo").on("click", function () {
        // todo ask how the not might work, because it seems like it needs another logical operator to be applied on
        // todo ask about whether I need to fix that the rate will return different results as to when it is applied?
        // todo ask how I might start saving certain overlays of results to the datagroup
        // todo ask how changes to the data appearance might make this different (say columns don't have the same number of entries)
        // so I could make a function that gets called on each of the bars that have been selected yes?
        function thresholdFilterSize(cir, set) {
            var radius = parseFloat(cir.attributes.r.value);
            if (radius < sizeScale(criteria.threshold.value)) {
                set.add(cir.__data__.data[0]); // why does having workingData as an argument make it in the local scope all of a sudden?
            }
        }

        function thresholdFilterColor(cir, set) { //todo ask about how to easily give users a way to select a reasonable value, because numbers are really small
            // will I need to be able to parse exponential(scientific) digits?
            // should I give people a sampling tool for the color? like an eyedropper?
            var nodeInfo = PSE_nodesInfo[cir.__data__.data[0][0]][cir.__data__.data[0][1]];
            if (nodeInfo.color_weight < criteria.threshold.value) {
                set.add(cir.__data__.data[0]);
            }
        }

        function rateFilterColor(cir, set) {
            allCircles // why does putting this here actually create a reference to it?
            var focusRow = cir.__data__.data[0][1], //zero based index
                focusCol = cir.__data__.data[0][0],
                colorWeight = PSE_nodesInfo[focusCol][focusRow].color_weight, //essentially the same as the row above
                topRow = PSE_nodesInfo[0].length - 1, //1 based index so subtract 1 PERHAPS NOT NEEDED INSIDE FUNCTION AS VALUES ARE STATIC
                rightCol = PSE_nodesInfo.length - 1;
            if (focusRow != topRow && focusCol != rightCol) { //wha is a good algorithmic way to check all these options efficiently
                var vertCircle = allCircles[0][focusRow + focusCol + 1],//allcircles is ordered in columns, so this selects dot above
                    horzCircle = allCircles[0][focusCol + focusRow + 3];
                // what will happen if we start making non linear groups here?
            }
            else if (focusRow == topRow && focusCol != rightCol) {
                var vertCircle = allCircles[0][focusRow + focusCol - 1],
                    horzCircle = allCircles[0][focusCol + focusRow + 3]
            }
            else if (focusRow != topRow && focusCol == rightCol) {
                var vertCircle = allCircles[0][focusRow + focusCol + 1],
                    horzCircle = allCircles[0][focusCol + focusRow - 3]

            }
            else {
                var vertCircle = allCircles[0][focusRow + focusCol - 1],
                    horzCircle = allCircles[0][focusCol + focusRow - 3]
            }


            for (otherCir of [horzCircle, vertCircle]) { // todo determine how to parse scientific entries
                var otherCirColWeight = PSE_nodesInfo[otherCir.__data__.data[0][0]][otherCir.__data__.data[0][1]].color_weight,  //the inverting brings out values that are related to max and min for the size
                    colDiff = Math.abs(colorWeight - otherCirColWeight);
                if (colDiff < criteria.rate.value) {
                    set.add(cir.__data__.data[0])
                }

            }
        }

        function rateFilterSize(cir, set) {
            //todo come back and fix the indexing so that the vertical circles are actually the ones that are being selected in col numbers greater than 0
            allCircles, sizeScale; // why does putting this here actually create a reference to it?
            var focusRow = cir.__data__.data[0][1], //zero based index
                focusCol = cir.__data__.data[0][0] * 3, // the multiplication by 3 should be a necessary conversion number to match with the indices of the actual sized array.t
                topRow = PSE_nodesInfo[0].length - 1, //1 based index so subtract 1 PERHAPS NOT NEEDED INSIDE FUNCTION AS VALUES ARE STATIC
                rightCol = PSE_nodesInfo.length - 1;
            if (focusRow != topRow && focusCol / 3 != rightCol) { //wha is a good algorithmic way to check all these options efficiently
                var vertCircle = allCircles[0][focusRow + focusCol + 1],//allcircles is ordered in columns, so this selects dot above
                    horzCircle = allCircles[0][focusCol + focusRow + 3];
                // what will happen if we start making non linear groups here?
            }
            else if (focusRow == topRow && focusCol / 3 != rightCol) {
                var vertCircle = allCircles[0][focusRow + focusCol - 1],
                    horzCircle = allCircles[0][focusCol + focusRow + 3]
            }
            else if (focusRow != topRow && focusCol == rightCol) {
                var vertCircle = allCircles[0][focusRow + focusCol + 1],
                    horzCircle = allCircles[0][focusCol + focusRow - 3]

            }
            else {
                var vertCircle = allCircles[0][focusRow + focusCol - 1],
                    horzCircle = allCircles[0][focusCol + focusRow - 3]
            }
            ;


            for (otherCir of [horzCircle, vertCircle]) { //todo have conditional check to prevent the top row and the far right from making duplicates
                var radDiff = Math.abs(sizeScale.invert(cir.attributes.r.value) - sizeScale.invert(otherCir.attributes.r.value)),
                    lineFunc = d3.svg.line()
                        .x(function (d) {
                            return d.x
                        })
                        .y(function (d) {
                            return d.y
                        })
                        .interpolate("linear");

                if (radDiff > criteria.rate.value) { // the < should make it so that only items with large changes of metric are kept on canvas
                    // set.add(cir.__data__.data[0])
                    //todo still make things sensitive to placement on the canvas, (border cases)
                    //todo figure out what's happening in the bottom right of the canvas
                    var cirRad = +cir.attributes.r.value,
                        cirX = +cir.attributes.cx.value,
                        cirY = +cir.attributes.cy.value,
                        otherRad = +otherCir.attributes.r.value,
                        otherX = +otherCir.attributes.cx.value,
                        otherY = +otherCir.attributes.cy.value,
                        diffDistX = ((otherX + otherRad) - (cirX + cirRad)) / 2,
                        diffDistY = ((cirY + cirRad) - (otherY + otherRad)) / 2,
                        lineData = [];
                    if (cirX - otherX == 0) { //determines which pair we are examining, if zero it is vert circle
                        var lineData = [{
                            y: cirY - diffDistY, // this is the bottom position of the focused circle
                            x: cirX - cirRad
                        },
                            {
                                y: cirY - diffDistY, // this is the bottom position of the focused circle
                                x: cirX + cirRad
                            }]

                    } else {
                        // this should calculate the distance between the inner edges of the circles and then divide by 2
                        var lineData = [{
                            x: cirX + diffDistX, // this is the bottom position of the focused circle
                            y: cirY + cirRad
                        },
                            {
                                x: cirX + diffDistX, // this is the bottom position of the focused circle
                                y: cirY - cirRad
                            }]
                    }
                    ;
                    if (lineData != null) {
                        d3.select(".dotsCanvas").append("path")
                            .attr("d", lineFunc(lineData))
                            .attr("stroke", radDiffColScale(radDiff))
                            .attr("stroke-width", "2px")
                            .attr("fill", "none");
                    }
                }
            }
        }


        var allCircles = d3.selectAll("circle"),
            min_size = d3.select("#minShapeLabel").node().innerHTML,
            max_size = d3.select("#maxShapeLabel").node().innerHTML,
            sizeScale = d3.scale.linear()
                .domain([+min_size, +max_size]) // these plus signs convert the string to number
                .range(d3.extent(workingData, function (d) {
                    return +d.points.radius
                })), // makes sure that we don't start creating negative radii based on user input, clamps to upper or lower bounds
            criteria = {
                threshold: {//currently this is hard coded for the size filters which needs to be updated
                    value: +d3.select("#threshold").node().value
                    , type: d3.select("input[name=threshold]:checked").node().id
                },//specifies color versus size measurements
                rate: { // how to relate rate of change  to the max and min
                    value: +d3.select("#rateOfChange").node().value //in theory this won't need to have the scale, because the differences will be arbitrary value
                    , type: d3.select("input[name=rateOfChange]:checked").node().id
                },
                logic: d3.select("input[name=logicButton]:checked").node().id
            },
            removalSet,
            radDiffColScale = d3.scale.linear()
                .domain([0, max_size - min_size])
                .range(["white", "red"]);

        if (criteria.logic == "Or") {
            var thresholdSet = new Set(),
                rateSet = new Set();
            d3.selectAll("circle")[0].forEach(function (d) {// [0] part seems strange, is there another way to use forEach without it?
                if (criteria.threshold.type == "Size" && criteria.rate.type == "Size") {
                    thresholdFilterSize(d, thresholdSet);
                    rateFilterSize(d, rateSet);

                } else if (criteria.threshold.type == "Color" && criteria.rate.type == "Size") {
                    thresholdFilterColor(d, thresholdSet);
                    rateFilterSize(d, rateSet);

                } else if (criteria.threshold.type == "Size" && criteria.rate.type == "Color") {
                    thresholdFilterSize(d, thresholdSet);
                    rateFilterColor(d, rateSet);
                } else {
                    thresholdFilterColor(d, thresholdSet)
                    rateFilterColor(d, rateSet)
                }
            });
            // @formatter:off
            removalSet = new Set([...thresholdSet, ...rateSet]);// this performs a union of the two sets, and the actual syntax is messing up pycharm
            // @formatter: on
        } else if (criteria.logic == "And") {
            var thresholdSet = new Set(),
                rateSet = new Set();
            d3.selectAll("circle")[0].forEach(function (d) {// [0] part seems strange, is there another way to use forEach without it?
                if (criteria.threshold.type == "Size" && criteria.rate.type == "Size") {
                    thresholdFilterSize(d, thresholdSet);
                    rateFilterSize(d, rateSet);

                } else if (criteria.threshold.type == "Color" && criteria.rate.type == "Size") {
                    thresholdFilterColor(d, thresholdSet);
                    rateFilterSize(d, rateSet);

                } else if (criteria.threshold.type == "Size" && criteria.rate.type == "Color") {
                    thresholdFilterSize(d, thresholdSet);
                    rateFilterColor(d, rateSet);
                } else {
                    thresholdFilterColor(d, thresholdSet)
                    rateFilterColor(d, rateSet)
                }

            });

            //line below is pycharm commmand to prevent bug triggered by auto format
            // @formatter:off
            removalSet = new Set([...thresholdSet].filter(x => rateSet.has(x))) // this is an intersection for set arithmetic the [...] converts by spreading out elements => is a shorthand function form
            // @formatter:on


        }
        removalSet.forEach(function (indPair) {
            workingDataRemove(indPair, workingData)
        });
        transparentDots()
    });

d3.select("#addFilterOps").on("click", function (d) {
    console.log("")
});


    d3.selectAll("circle").on("mouseover", function (d) {
        var nodeInfo = PSE_nodesInfo[d.data[0][0]][d.data[0][1]];
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
    d3.selectAll("circle").on("click", function (d) {
        var nodeInfo = PSE_nodesInfo[d.data[0][0]][d.data[0][1]];
        if (nodeInfo.dataType != undefined) {
            displayNodeDetails(nodeInfo['Gid'], nodeInfo['dataType'], pageParam); // curious because backPage isn't in the scope, but appears to work.
        }
    })

}
/*
 * Do a redraw of the plot. Be sure to keep the resizable margin elements as the plot method seems to destroy them.
 */
function redrawPlot(plotCanvasId) {
    /*// todo: mh the selected element is not an ancestor of the second tab!!!
     // thus this redraw call fails, ex on resize

     if (_PSE_plot != null) {
     _PSE_plot = $.plot($('#' + plotCanvasId)[0], _PSE_plot.getData(), $.extend(true, {}, _PSE_plotOptions));
     }*/
    //it appears that there is a tie in for window.resize to this function. Lets see how this works out
    if (backPage == null || backPage == '') {
        var backPage = get_URL_param('back_page');
    }
    PSE_mainDraw('main_div_pse', backPage)

}


/*
 * Fire DataType overlay when clicking on a node in PSE.
 */
function applyClickEvent(canvasId, backPage) {
    var currentCanvas = $("#" + canvasId);
    currentCanvas.unbind("plotclick");
    currentCanvas.bind("plotclick", function (event, pos, item) {
        if (item != null) {
            var dataPoint = item.datapoint;
            var dataInfo = PSE_nodesInfo[dataPoint[0]][dataPoint[1]];
            if (dataInfo['dataType'] != undefined) {
                displayNodeDetails(dataInfo['Gid'], dataInfo['dataType'], backPage);
            }
        }
    });
}

var previousPoint = null;
/*
 * On hover display few additional information about this node.
 */
function applyHoverEvent(canvasId) {
    $("#" + canvasId).bind("plothover", function (event, pos, item) {
        if (item) {
            if (previousPoint != item.dataIndex) {
                previousPoint = item.dataIndex;
                $("#tooltip").remove();
                var dataPoint = item.datapoint;
                var dataInfo = PSE_nodesInfo[dataPoint[0]][dataPoint[1]];
                var tooltipText = ("" + dataInfo["tooltip"]).split("&amp;").join("&").split("&lt;").join("<").split("&gt;").join(">");

                $('<div id="tooltip"> </div>').html(tooltipText
                ).css({
                        position: 'absolute', display: 'none', top: item.pageY + 5, left: item.pageX + 5,
                        border: '1px solid #fdd', padding: '2px', 'background-color': '#C0C0C0', opacity: 0.80
                    }
                ).appendTo('body').fadeIn(200);
            }
        } else {
            $("#tooltip").remove();
            previousPoint = null;
        }
    });
}


function PSEDiscreteInitialize(labelsXJson, labelsYJson, series_array, dataJson, backPage, hasStartedOperations,
                               min_color, max_color, min_size, max_size) {


    var labels_x = $.parseJSON(labelsXJson);
    var labels_y = $.parseJSON(labelsYJson);
    var data = $.parseJSON(dataJson);

    min_color = parseFloat(min_color);
    max_color = parseFloat(max_color);
    min_size = parseFloat(min_size);
    max_size = parseFloat(max_size);

    if (d3.select("#control-view").empty() != true) {
        ColSch_initColorSchemeGUI(min_color, max_color, function () {
            _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage);
        });
    }

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
    _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage); // why is this called a second time?


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


/**
 * Changes the series colors according to the color picker component.
 */
function changeColors() {
    var series = _PSE_plot.getData();
    for (var i = 0; i < series.length; i++) {
        var indexes = series[i].datapoints.points;
        var dataInfo = PSE_nodesInfo[indexes[0]][indexes[1]];
        var colorWeight = dataInfo['color_weight'];
        var color = ColSch_getGradientColorString(colorWeight, _PSE_minColor, _PSE_maxColor);
        series[i].points.fillColor = color;
        series[i].color = color;
    }
    _PSE_plot.draw();
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


