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
/* global doAjaxCall, displayMessage */


// We keep all-nodes information for current PSE as a global, to have them ready at node-selection, node-overlay.
var _PSE_d3NodesInfo;
var _PSE_seriesArray;
var _PSE_hasStartedOperations;

// Keep Plot-options and MIN/MAx colors for redraw (e.g. at resize).
var _PSE_plotOptions;
var _PSE_minColor;
var _PSE_maxColor;
var _PSE_plot;
var _PSE_back_page;
var _PSE_hasDatatypeMeasure=true;

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

function d3Plot(placeHolder, data, options, pageParam) {

    /*************************************************************************************************************************
     *   what follows here first are functions meant to assist in the plotting of results correctly with d3 as our plotting language.
     *************************************************************************************************************************/


    //these lines are cleanup for artifacts of the conversion that aren't behaving nicely, they should eventually be removed because they are just treating the symptoms.
    if (d3.select(".outerCanvas").empty() != true) {
        d3.selectAll(".outerCanvas").remove()
    }
    if (d3.selectAll("#main_div_pse")[0].length > 1) {
        var oneOrMoreDiv = d3.selectAll("div > div.flex-wrapper"); //index necessary because selection is an array with two elements, and second is unneccessary

        if (oneOrMoreDiv[0].length > 1) {
            oneOrMoreDiv[0][1].remove();
        } else {
            oneOrMoreDiv[0][0].remove();
        }
    }

    /*
     * The whole point of this function is to make the graph appear less crowded towards the axes, and each other dot. The point of discriminating between x and y is stylistic choice.
     */
    function createScale(xORy, labelArr) {
        var lowerExtent, upperExtent, extentPadding, padLo, padUp, newScale;
        if (xORy === "x") {
            [lowerExtent, upperExtent] = d3.extent(labelArr);
            extentPadding = ((upperExtent - lowerExtent) * .10) / 2; // this multiplication factor controls how much the dots are gathered together
            [padLo, padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];

            newScale = d3.scale.linear()
                .domain([padLo, padUp])
                .range([_PSE_plotOptions.margins.left, innerWidth - _PSE_plotOptions.margins.right]);

        } else {
            [lowerExtent, upperExtent] = d3.extent(labelArr);
            extentPadding = (Math.max(upperExtent - lowerExtent, 1) * .35) / 2;
            [padLo, padUp] = [lowerExtent - extentPadding, upperExtent + extentPadding];

            newScale = d3.scale.linear()
                .domain([padLo, padUp])
                .range([innerHeight - (_PSE_plotOptions.margins.bottom), _PSE_plotOptions.margins.top]);
        }
        return newScale;
    }

    /*
     * This functions generate an x or y axis structure with floating point accuracy up to 3 decimals. This will be used in slightly varying ways for the grid lines, and
     * the typical axes of the graph.
     */
    function createXAxis() {

        newAxis = d3.svg.axis().scale(xScale)
            .orient("bottom")
            .tickValues(_PSE_plotOptions.xaxis.values)
            .tickFormat(function (i) {
                current_idx = _PSE_plotOptions.xaxis.values.indexOf(i);
                if (typeof _PSE_plotOptions.xaxis.labels[current_idx] === "string") {
                    return _PSE_plotOptions.xaxis.labels[i]
                } else {
                    return d3.format(",.3f")(i)
                }

            });
        return newAxis
    }

    function createYAxis() {
        newAxis = d3.svg.axis().scale(yScale)
            .orient("left")
            .tickValues(_PSE_plotOptions.yaxis.values)
            .tickFormat(function (i) {
                current_idx = _PSE_plotOptions.yaxis.values.indexOf(i);
                if (typeof _PSE_plotOptions.yaxis.labels[current_idx] === "string") {
                    return _PSE_plotOptions.yaxis.labels[i]
                } else {
                    return d3.format(",.3f")(i)
                }

            });
        return newAxis
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
                for (let i = 0; i < d3.selectAll(".action-store")[0].length; i++) { // note the indexing due to the selectAll returning a one ele array of multiple arrays
                    const selectElement = d3.select("#filterSelect" + i);
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
     * This function is what helps coordinate the new positions of each of our result dots when the user pans the graph, or zooms in and out.
     * the factor section prevents the dots from overlapping each other after extreme amounts of zooming, but it makes it possible to scale up
     * small sections of a great big batch result.
     */
    function moveDots() {
        circles
            .transition()
            .attr({
                r: function (d) {
                    const factor = xzoom.scale() * yzoom.scale();
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

        canvas.append("g").attr("id", "brushLines");
        var lineData;

        for (let xVal of d3.range(span[0][0], span[1][0], steps[0])) {
            lineData = [{x: xScale(xVal), y: yScale(span[0][1])}, {x: xScale(xVal), y: yScale(span[1][1])}];
            d3.select("#brushLines").append("path")
                .attr("d", lineFunc(lineData))
                .attr("stroke", "blue")
                .attr("stroke-width", ".5px")
                .attr("fill-opacity", ".1")
                .attr("fill", "none")
                .attr("id", "brushLine");
        }
        for (let yVal of d3.range(span[0][1], span[1][1], steps[1])) {
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

        const colTest = ColSch_getGradientColorString(weight, _PSE_minColor, _PSE_maxColor).replace("a", "");
        // the a creates an error in the color scale creation, so it must be removed.
        // turn color string into a d3 compatible form.
        return d3.rgb(colTest)
    }


    /*******************************************************************************
     *  this section below is devoted to variable declaration, and graph assembly.
     ******************************************************************************/

    let myBase, workingData, canvasDimensions, canvas, xScale, yScale, xAxis, yAxis, circles, brush,
        dotsCanvas, innerHeight, innerWidth, toolTipDiv, zoom, datatypeGID, structure, inclusiveX, inclusiveY, steps;
    myBase = d3.select(placeHolder);
    workingData = sortResults(data); //structure expects the sorting that this function performs.
    [inclusiveX, inclusiveY] = constructLabels(workingData);
    steps = {
        x: [+(inclusiveX[1] - inclusiveX[0]).toFixed(4)] // prevent any wierd float point arithmetic
        , y: [+(inclusiveY[1] - inclusiveY[0]).toFixed(4)]
    };
    structure = createStructure(workingData, inclusiveX, inclusiveY);
    for (let ind in workingData) {
        workingData[ind].key = parseFloat(ind)
    }

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
        .attr("x", -_PSE_plotOptions.margins.left)// these two areas are simply selected for what they accomplish visually. I wonder if there could be a real connection to the values used for arranging the canvas
        .attr("y", _PSE_plotOptions.margins.top)
        .attr("width", _PSE_plotOptions.margins.right)//
        .attr("height", innerHeight - _PSE_plotOptions.margins.bottom - _PSE_plotOptions.margins.top);

    toolTipDiv = d3.select(".tooltip"); // this is the hover over dot tool tip which displays information stored in the PSE_nodeInfo variable.
    xAxis = createXAxis(); // these lines create the axes and grids
    xGrid = createXAxis()
        .tickSize(innerHeight, 0, 0)
        .tickFormat(""); //means don't show any values at the base line of the axis
    yAxis = createYAxis();
    yGrid = createYAxis()
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
                return xScale(d.coords.x); // the xScale converts from data values to pixel values for appropriate actual placement in the graphed space.
            },
            cy: function (d) {
                return yScale(d.coords.y)
            },
            fill: function (d) {
                let color = d3.rgb("black"); // leave pending results blacked out if not complete.
                let nodeInfo = getNodeInfo(d.coords);
                if (nodeInfo.tooltip.search("PENDING") === -1 && nodeInfo.tooltip.search("CANCELED") === -1) { // this prevents code from trying to find reasonable color values when simulation results haven't been generated for them
                        color = returnfill(nodeInfo.color_weight);
                }
                return color; // otherwise fill out with color in keeping with scheme.
            }
        });


    /*************************************************************************************************************************
     *   what follows here is the associated code for mouse events, and user generated events
     *************************************************************************************************************************/

    /*
     * specifics for drawing the explore menu upon clicking the explore button, and the generation of brush for the graph.
     */
    d3.select("#Explore").on("click", function () {
        var exploreDiv = d3.select("#exploreDiv");
        if (exploreDiv.style("display") === "none") {
            exploreDiv.style("display", "block");

        } else {
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

            lineFillBrush(extent, [xSteps, ySteps]); // draw initial grid lines when mouse button is released.
            var elemSliderA = $('#XStepSlider'); // this is included here to make the sliders affect the drawing of the grid lines dynamically
            elemSliderA.slider({
                min: 0, max: extent[1][0] - extent[0][0], step: .0001, value: xSteps,
                slide: function (event, ui) {
                    xSteps = ui.value;
                    d3.select('input[name="xStepInput"]').property('value', ui.value);
                    d3.select("#brushLines").remove(); // remove the previously drawn lines to prevent confusion.
                    lineFillBrush(extent, [xSteps, ySteps]); // redraw grid lines
                }
            });
            var elemSliderB = $('#YStepSlider');
            elemSliderB.slider({
                min: 0, max: extent[1][1] - extent[0][1], step: .0001, value: ySteps,
                slide: function (event, ui) {
                    ySteps = ui.value;
                    d3.select('input[name="yStepInput"]').property('value', ui.value);
                    d3.select("#brushLines").remove();
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

        var filterDiv = d3.select("#FilterDiv");
        if (filterDiv.style("display") == "none") {
            filterDiv.style("display", "block");
            getFilterSelections();
            refreshOnChange()

        } else {
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
                var thresholdValue = +groupSelection.select(".thresholdInput").property('value'),
                    notPreference = (groupSelection.select("input[name='notButton']").property('checked')) ? "<" : ">", //gives us true or false
                    filterType = groupSelection.select("input[name='thresholdType']:checked").property('id').slice(0, -1); // gives either 'Color' or 'Size'
                concatStr += logicalOperator + ' ' + filterType + ' ' + notPreference + ' ' + thresholdValue;

            }

            return concatStr;
        }

        var min_size = +d3.select("#minShapeLabel").node().innerHTML,
            max_size = +d3.select("#maxShapeLabel").node().innerHTML,
            sizeScale = d3.scale.linear()
                .range([min_size, max_size]) // these plus signs convert the string to number
                .domain(d3.extent(workingData, function (d) {
                    return +d.points.radius
                }));

        for (var circle of d3.selectAll('circle')[0]) {
            var radius = sizeScale(+circle.attributes.r.value),
                filterString = concatCriteria(),
                data = circle.__data__,
                colorWeight = getNodeInfo(data.coords).color_weight;

            filterString = filterString.replace(/Size/g, radius).replace(/Color/g, colorWeight);
            if (!eval(filterString)) { // this phrasing is now persistent phrased, meaning that the data that the user wants to keep doesn't pass.
                var repInd = workingData.indexOf(data);
                if (repInd != -1) { // keeps the function from repeatedly detecting transparent dots, but removing others from workingData that pass the criteria
                    workingData.splice(repInd, 1); //this will remove the data from the group, and then it can be selected for below in the transparent dots
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
                d3.select("#FilterDiv > ul").append("li").html(r);
                getFilterSelections();
                refreshOnChange()
            },
            error: function () {
                displayMessage("couldn't add new row of filter options", "errorMessage")
            }
        })
    });

    function getNodeInfo(coords) {
        var res;
        try {
            res = _PSE_d3NodesInfo[coords.x][coords.y];
            if (res == undefined) {
                throw Exception()
            }
            return res;
        } catch (err) {
            try {
                res = _PSE_d3NodesInfo[coords.x.toFixed(1)][coords.y];
                if (res == undefined) {
                    throw Exception()
                }
                return res;
            } catch (err) {
                try {
                    res = _PSE_d3NodesInfo[coords.x][coords.y.toFixed(1)];
                    if (res == undefined) {
                        throw Exception()
                    }
                    return res;
                } catch (err) {
                    try {
                        res = _PSE_d3NodesInfo[coords.x.toFixed(1)][coords.y.toFixed(1)];
                        if (res == undefined) {
                            throw Exception()
                        }
                        return res;
                    } catch (err) {
                        displayMessage(err, "errorMessage");
                    }
                }
            }
        }
    }

    /*
     * provides basis for behavior when the cursor passes over a result. we get shown the message that is stored as information in the nodeInfo.tooltip
     */
    d3.selectAll("circle")
        .on("mouseover", function (d) {
            let offsetX = window.event.pageX;
            let offsetY = window.event.pageY - 100;
            const pseContainer = document.getElementById("section-pse");
            if (pseContainer) {
                let relativeOffsetLeft = pseContainer.offsetLeft;
                let relativeOffsetTop = pseContainer.offsetTop;
                offsetX = offsetX - relativeOffsetLeft;
                offsetY = offsetY - relativeOffsetTop;
            }
            let nodeInfo = getNodeInfo(d.coords);
            let toolTipText = nodeInfo.tooltip.split("&amp;").join("&").split("&lt;").join("<").split("&gt;").join(">");
            toolTipDiv.html(toolTipText);
            toolTipDiv.style({
                position: "absolute",
                left: (offsetX) + "px",
                top: (offsetY) + "px",
                display: "block",
                'background-color': '#C0C0C0',
                border: '1px solid #fdd',
                padding: '2px',
                opacity: 0.80,
                'z-index': 999
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
        const nodeInfo = getNodeInfo(d.coords);
        if (nodeInfo.dataType !== undefined) {
            displayNodeDetails(nodeInfo['Gid'], nodeInfo['dataType'], pageParam); // curious because backPage isn't in the scope, but appears to work.
        }
    });


    /*
     * This is the function that specifies how to convert the graph into an exportable file.
     */
    d3.select("#ctrl-action-export").on("click", function (d) {

    })
}


function PSEDiscreet_Initialize(labelsXJson, labelsYJson, valuesXJson, valuesYJson, dataJson, backPage,
                                hasStartedOperations, min_color, max_color, min_size, max_size) {

    var labels_x = $.parseJSON(labelsXJson);
    var labels_y = $.parseJSON(labelsYJson);
    var values_x = $.parseJSON(valuesXJson);
    var values_y = $.parseJSON(valuesYJson);
    var data = $.parseJSON(dataJson);
    _PSE_back_page = backPage
    min_color = parseFloat(min_color);
    max_color = parseFloat(max_color);
    min_size = parseFloat(min_size);
    max_size = parseFloat(max_size);

    ColSch_initColorSchemeGUI(min_color, max_color, function () {
         updateLegend2D(min_color,max_color);
         d3Plot('#main_div_pse', _PSE_seriesArray, $.extend(true, {}, _PSE_plotOptions), _PSE_back_page);
    });

    function _fmt_lbl(sel, v) {
        $(sel).html(Number.isNaN(v) ? 'not available' : toSignificantDigits(v, 3));
    }

    _fmt_lbl('#minColorLabel', min_color);
    _fmt_lbl('#maxColorLabel', max_color);
    _fmt_lbl('#minShapeLabel', min_size);
    _fmt_lbl('#maxShapeLabel', max_size);

    // updateLegend with min_color and max_color
    updateLegend2D(min_color, max_color);

    if (Number.isNaN(min_color)) {
        min_color = 0;
        max_color = 1;
        _PSE_hasDatatypeMeasure=false;
    }
    _updatePlotPSE('main_div_pse', labels_x, labels_y, values_x, values_y, min_color, max_color, backPage, data);
}


function _updatePlotPSE(canvasId, xLabels, yLabels, xValues, yValues, min_color, max_color, backPage, d3Data) {

    // why is it that we don't associate the labels into the attributes of the actual data_info or seriesArray?
    // where does the seriesArray structure get created?
    _PSE_minColor = min_color;
    _PSE_maxColor = max_color;
    _PSE_d3NodesInfo = d3Data;
    _PSE_plotOptions = {
        margins: { // is this the correct way to be doing margins? It's just how I have in the past,
            top: 20,
            bottom: 40,
            left: 50,
            right: 50
        },
        xaxis: {
            labels: xLabels,
            values: xValues
        },
        yaxis: {
            labels: yLabels,
            values: yValues
        }

    };
}


function PSEDiscreet_BurstDraw(parametersCanvasId, backPage, groupGID) {

    if (groupGID === null || groupGID === undefined) {
        // We didn't get parameter, so try to get group id from page
        groupGID = document.getElementById("datatype-group-gid").value;
    }
    if (backPage === undefined || backPage === null || backPage === '') {
        backPage = get_URL_param('back_page');
    }

    const selectedColorMetric = $('#color_metric_select').val();
    const selectedSizeMetric = $('#size_metric_select').val();
    const url = '/burst/explore/draw_discrete_exploration/' + groupGID + '/' + backPage +
                '/' + selectedColorMetric + '/' + selectedSizeMetric;

    doAjaxCall({
        type: "POST",
        url: url,
        success: function (r) {
            $('#' + parametersCanvasId).html(r);
        },
        error: function () {
            displayMessage("Could not refresh with the new metrics.", "warningMessage");
        }
    });
}


function PSEDiscreet_RedrawResize(){
    d3Plot('#main_div_pse', _PSE_seriesArray, $.extend(true, {}, _PSE_plotOptions), _PSE_back_page);

    if (_PSE_hasStartedOperations) {
        setTimeout("PSEDiscreet_LoadNodesMatrix()", 3000);
    }
}


function PSEDiscreet_LoadNodesMatrix(groupGID) {
    if (groupGID === null || groupGID === undefined) {
        groupGID = document.getElementById("datatype-group-gid").value;
    }
    const selectedColorMetric = $('#color_metric_select').val();
    const selectedSizeMetric = $('#size_metric_select').val();

    doAjaxCall({
        url: '/burst/explore/get_series_array_discrete/',
        data: {'datatype_group_gid': groupGID,
               'back_page':_PSE_back_page,
               'color_metric': selectedColorMetric,
               'size_metric': selectedSizeMetric},
        type: 'GET',
        async: false,
        success: function (data) {
            let pse_context=$.parseJSON(data);
            _PSE_hasStartedOperations=pse_context.has_started_ops;
            _PSE_seriesArray = $.parseJSON(pse_context.series_array);
            PSEDiscreet_RedrawResize();
        }
    });
}

function updateLegend2D(minColor, maxColor) {
    let legendContainer, legendHeight, tableContainer;
    legendContainer = d3.select("#colorWeightsLegend");
    legendHeight = legendContainer.node().getBoundingClientRect().height;
    tableContainer = d3.select("#table-colorWeightsLegend");
    ColSch_updateLegendColors(legendContainer.node(), legendHeight);
    if(!_PSE_hasDatatypeMeasure){
        minColor=NaN;
        maxColor=NaN;
    }
    ColSch_updateLegendLabels(tableContainer.node(), minColor, maxColor, legendHeight);
}
