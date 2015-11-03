/* globals d3 */

// ====================================    INITIALIZATION CODE START =========================================
(function(){ // module timeseriesFragment
var tsFrag = {
    dataTimeSeries: "",                 // Contains the address to query the time series of a specific voxel.
    m: [],                              // Global preview margin sizes
    w: 0,                               // Global preview width
    h: 0,                               // Global preview width
    smallMargin: {},                    // Sortable time series margins
    width: 0,                           // Sortable time series width
    height: 0,                          // Sortable time series height
    tsDataArray: [],                    // Array containing all the time series data
    selectedIndex: 0,                   // Selected time series line index
    relevantSortingFeature: "mean",     // Stores what feature to consider while sorting the time series.
    relevantColoringFeature: "mean",    // Stores what feature to consider while coloring the time series.
    timeLength: 0,                      // The length of the data in frames
    samplePeriod: 0,                    // Meta data. The sampling period of the time series
    samplePeriodUnit: "",               // Meta data. The time unit of the sample period
    selectedEntity: [],                 // The selected voxel; [i, j, k].
    line: null,                         // A d3.line() for the global graph.
    sortableline: null,                 // A d3.line() for the sortable graph.
    x: null,                            // A d3.scale() for the x axis
    y: null,                            // A d3.scale() for the y axis
    sortableY: null,                    // Y axis labels for the sortable graph
    xAxisScale:null,                    // X axis labels for the global graph
    brush: null                         // A d3.brush()
};

/**
 * We can pass just the time series array as the second element or an Object
 * with all the relevant data, including the array itself.
 */
var tsDataObj = function(params, dataArray) {             // this keeps all the data about a specific time series
    this.x = params.x || tsFrag.selectedEntity[0];                      // The X coordinate
    this.y =  params.y || tsFrag.selectedEntity[1];                     // The Y coordinate
    this.z = params.z || tsFrag.selectedEntity[2];                      // The Z coordinate
    this.label = params.label || "["+this.x+","+this.y+","+this.z+"]";  // The string used as a label for this Object
    this.data = params.data || dataArray;                               // The actual data
    this.max = params.max || d3.max(this.data);                         // The maximum value on the data
    this.min = params.min || d3.min(this.data);                         // The minimum value on the data
    this.mean = params.mean || d3.mean(this.data);                      // The mean values on the data
    this.median = params.median || d3.median(this.data);                // The median value on the data
    this.variance = params.variance || variance(this.data, this.mean);  // The variance of the data
    this.deviation = params.deviation || Math.sqrt(this.variance);       // The std deviation of the data
};

/**
 * Make all the necessary initialisations
 * @param tsDataRequest URLs of the dataset we're working on
 */
function TSF_initVisualizer(tsDataRequest, timeLength, samplePeriod, samplePeriodUnit, minimumValue, maximumValue){
    tsFrag.urlTSData = tsDataRequest;

    // take all the necessary data from the time series volume visualizer
    tsFrag.timeLength = timeLength;
    tsFrag.samplePeriod = samplePeriod;
    tsFrag.samplePeriodUnit = samplePeriodUnit;
    tsFrag.minimumValue = minimumValue;
    tsFrag.maximumValue = maximumValue;

    // tsFrag own data
    // define dimensions of graph
    var graphWidth = $('#graph').width();
    tsFrag.m = [30, 80, 30, 80]; // margins
    tsFrag.smallMargin = {top: 0, right: 80, bottom: 0, left: 80};
    tsFrag.width = graphWidth - tsFrag.smallMargin.left - tsFrag.smallMargin.right;
    tsFrag.height = 30 - tsFrag.smallMargin.top - tsFrag.smallMargin.bottom;
    tsFrag.w = graphWidth - tsFrag.m[1] - tsFrag.m[3]; // width
    tsFrag.h = 240 - tsFrag.m[0] - tsFrag.m[2]; // height

    attachUIListeners();
}

/**
 * Update the variables that this fragment shares with other visualizers
 */
function updateTSFragment(selectedEntity, currentTimePoint){
    tsFrag.selectedEntity = selectedEntity;
    tsFrag.currentTimePoint = currentTimePoint;
}

// ====================================    INITIALIZATION CODE END   =========================================

// ====================================    DRAWING FUNCTIONS START ===========================================

/**
 *  Add the selected entity to te time series array if it is not present yet and draw all of the SVG graphs.
 */
function drawGraphs(){
    $('#graph').empty();

    var label = "["+tsFrag.selectedEntity[0]+","+tsFrag.selectedEntity[1]+","+tsFrag.selectedEntity[2]+"]";
    var selectedVoxelIsNotPresent = !tsFrag.tsDataArray.some(function(ts){ return ts.label === this[0];}, [label]);

    if( selectedVoxelIsNotPresent ){
        var tmp = new tsDataObj(getPerVoxelTimeSeries(tsFrag.selectedEntity[0], tsFrag.selectedEntity[1], tsFrag.selectedEntity[2]));
        tsFrag.tsDataArray.push(tmp);
        var pvt = {x: tsFrag.selectedEntity[0], y:  tsFrag.selectedEntity[1],z:  tsFrag.selectedEntity[2]};
        sortTsGraphs($("#sortingSelector").val(), tsFrag.relevantSortingFeature, pvt);
    }
    if(tsFrag.tsDataArray.length < 1){
        return;
    }
    drawGobalTimeseries();
    updateBrush();
    drawSortableGraph();

    var miniContainer = $("#mini-container");
    if(miniContainer.children().length < 2) {
        miniContainer.sortable( "disable" );
        d3.selectAll("#mini-container g.list-item").classed("pin", true);
    }
}

/**
 * Draws the global graph showing all the selected time series at the same time.
 */
function drawGobalTimeseries(){
    tsFrag.x = d3.scale.linear().domain([0, tsFrag.timeLength]).range([0, tsFrag.w]);
    var x2   = d3.scale.linear().domain([0, tsFrag.timeLength]).range([0, tsFrag.w]);
    var y2 = d3.scale.linear().range([tsFrag.h, 0]);

    // Prepare the brush for later
    tsFrag.brush = d3.svg.brush()
        .x(x2)
        .on("brush", brush);

    function brush() {
        tsFrag.x.domain(tsFrag.brush.empty() ? x2.domain() : tsFrag.brush.extent());
        d3.select("#mini-container").selectAll(".tsv-line")
        .attr("d", function(d){
            tsFrag.sortableY = d3.scale.linear().domain([d.min, d.max]).range([tsFrag.height, 0]);
            return tsFrag.sortableline(d.data);
        });

       d3.select("#mini-container").selectAll(".tsv-x.tsv-axis").call(xAxis(tsFrag.height * tsFrag.tsDataArray.length));
       d3.select(".brusher").selectAll(".tsv-x.tsv-axis").call(xAxis(tsFrag.height));
    }

    var localMax = d3.max(tsFrag.tsDataArray, function(array){
      return array.max;
    });
    var localMin = d3.min(tsFrag.tsDataArray, function(array){
      return array.min;
    });
    tsFrag.y = d3.scale.linear().domain([localMin, localMax]).range([tsFrag.h, 0]);

    // create a line function that can convert data[] into x and y points
    tsFrag.line = d3.svg.line()
        .x(function(d,i){
            return tsFrag.x(i);
        })
        .y(function(d){
            return tsFrag.y(d);
        });

    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select("#graph").append("svg:svg")
          .attr("width", tsFrag.w + tsFrag.m[1] + tsFrag.m[3])
          .attr("height", tsFrag.h + tsFrag.m[0] + tsFrag.m[2] + tsFrag.smallMargin.top + tsFrag.smallMargin.bottom
                        + tsFrag.height * (tsFrag.tsDataArray.length + 2))
          .attr("class", "tsv-svg-component")
        .append("svg:g")
          .attr("transform", "translate(" + tsFrag.m[3] + "," + tsFrag.m[0] + ")");

    var rect = graph.append("rect")
        .attr('w',0)
        .attr('h',0)
        .attr('width', tsFrag.w)
        .attr('height', tsFrag.h)
        .attr('fill', "#ffffff")
        .attr("class", "graph-timeSeries-rect");

    // Add an overlay layer that will listen to most mouse events
    graph.append("rect")
        .attr("class", "overlay")
        .attr("width", tsFrag.w)
        .attr("height", tsFrag.h)
        .attr('fill', "#ffffff")
        .on("mouseover", function(){ focus.style("display", null); })
        .on("mouseout", function(){ focus.style("display", "none"); })
        .on("mousemove", TSF_mousemove);

    // create xAxis
    tsFrag.xAxisScale = d3.scale.linear().domain([0, tsFrag.timeLength*tsFrag.samplePeriod]).range([0, tsFrag.w]);
    var xAxis = function(lineHeight) {
        tsFrag.x.domain(tsFrag.brush.empty() ? x2.domain() : tsFrag.brush.extent());
        tsFrag.xAxisScale = tsFrag.x;
        tsFrag.xAxisScale.domain()[0] *= tsFrag.samplePeriod;
        tsFrag.xAxisScale.domain()[1] *= tsFrag.samplePeriod;
        return d3.svg.axis().scale(tsFrag.xAxisScale).tickSize(-lineHeight).tickSubdivide(true);
    };

    // Add the x-axis.
    graph.append("svg:g")
        .attr("class", "tsv-x tsv-axis")
        .attr("transform", "translate(0," + tsFrag.h + ")")
        .call(xAxis(tsFrag.h));

    // Add info about the time unit as a label
    graph.append("text")
        .attr("class", "tsv-x tsv-axis")
        .attr("text-anchor", "end")
        .attr("x", tsFrag.w + 60)
        .attr("y", tsFrag.h + 12)
        .text("Time (" + tsFrag.samplePeriodUnit + ")");

    // create left yAxis
    var yAxisLeft = d3.svg.axis().scale(tsFrag.y).ticks(4).orient("left");
    // Add the y-axis to the left
    graph.append("svg:g")
          .attr("class", "tsv-y tsv-axis")
          .attr("transform", "translate(-25,0)")
          .call(yAxisLeft);

    graph.append("text")
        .attr("class", "tsv-y tsv-axis")
        .attr("text-anchor", "end")
        .attr("x", "1em")
        .attr("y", "-2em")
        .attr("dy", ".75em")
        .text("Measurements");

    // Draw the time series lines on the main graph. Each one with it's own color
    graph.selectAll('.tsv-line')
        .data(tsFrag.tsDataArray)
        .enter()
        .append("path")
            .attr("class", "tsv-line")
            //.attr("clip-path", "url(#clip)")
            .attr("d", function(d){return tsFrag.line(d.data);})
            .attr('class', 'tsv-line tsv-colored-line')
            .attr("style", function(d){
                            return "stroke:" + ColSch_getAbsoluteGradientColorString(d[tsFrag.relevantColoringFeature]);} )
            .on("mouseover", selectLineData);

    // The focus will show the numeric value of a time series on a certain point
    var focus = graph.append("g")
        .attr("class", "focus")
        .style("display", "none");

    focus.append("circle")
        .attr("r", 4.5);

    focus.append("text")
        .attr("x", 9)
        .attr("dy", ".35em")
        .attr("style","background-color: aliceblue");
    var verticalLine = graph.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', tsFrag.h)
        .attr("stroke", "steelblue")
        .attr('class', 'verticalLine');

    // This red line will show what time point we are seeing in the main visualizer.
    var timeVerticalLine = graph.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', tsFrag.h)
        .attr("stroke", "red")
        .attr('class', 'timeVerticalLine')
        .attr("transform", function(){
                var width = $(".graph-timeSeries-rect").attr("width");
                var pos = (tsFrag.currentTimePoint*width)/(tsFrag.timeLength);
                return "translate(" + pos + ", 0)";
            });

    var circle = graph.append("circle")
        .attr("opacity", 0)
        .attr('r', 2)
        .attr("id", "mouseLineCircle");

    // Add listener so that the red line moves together with the main visualizer slider
    $("#movieSlider").on("slide change", function(){
        // Move blue line following the mouse
        var wdt = $(".graph-timeSeries-rect").attr("width");
        var xPos = (tsFrag.currentTimePoint*wdt)/(tsFrag.timeLength);

        var selectedLine = d3.select("path.tsv-highlight");
        if(selectedLine[0][0] == null){
            selectedLine = d3.select("path.tsv-colored-line:nth-of-type(1)")
        }
        var pathLength = selectedLine.node().getTotalLength();
        var beginning = xPos,
            end = pathLength,
            target;

        while(true){
            target = Math.floor((beginning + end) / 2);
            var pos = selectedLine.node().getPointAtLength(target);
            if((target === end || target === beginning) && pos.x !== xPos){
                break;
            }
            if(pos.x > xPos){
                end = target;
            }
            else if(pos.x < xPos){
                beginning = target;
            }
            else break; //position found
        }
    });

    var brusher = d3.select("#graph svg.tsv-svg-component").append("g")
            .attr("class", "brusher tsv-svg-component")
            .attr("width", tsFrag.width + tsFrag.smallMargin.left + tsFrag.smallMargin.right)
            .attr("height", tsFrag.height + tsFrag.smallMargin.top + tsFrag.smallMargin.bottom)
            .attr("id","brush")
            .attr("transform", "translate(" + tsFrag.smallMargin.left + "," + (tsFrag.smallMargin.top + tsFrag.h + tsFrag.m[0] + tsFrag.m[2]) + ")");

    brusher.append("rect")
        .attr('w',0)
        .attr('h',0)
        .attr("width", tsFrag.width)
        .attr("height", tsFrag.height)
        .attr('fill', "#ffffff")
        .style("stroke", "black")
        .attr("class", "brush-rect");

    // Align brush with other graphs
    brusher.append("g")
        .attr("class", "tsv-x tsv-axis")
        .attr("transform", "translate(0," + tsFrag.height + ")")
        .call(xAxis(tsFrag.height));

    // Add y label to brush
    brusher.append("text")
        .attr("class", "tsv-y label")
        .attr("text-anchor", "end")
        .attr("y", 6)
        .attr("dy", ".75em")
        .attr("transform", "translate(-5,0)")
        .text("Focus");

    // Add transparent overlay to display selection on brush
    brusher.append("g")
      .attr("class", "x tsv-brush")
      .call(tsFrag.brush)
    .selectAll("rect")
      .attr("y", -6)
      .attr("height", tsFrag.height + 6);
}

/**
 * Draws a sortable graph. Each line has the same vertical space as the others but the
 * scaling factor of each line is different. Lines are colored using the selected coloring feature
 * and they can be sorted manually or automatically by choosing one of many features.
 */
function drawSortableGraph(){
    tsFrag.sortableY = tsFrag.y;
    tsFrag.sortableline = d3.svg.line()
        .x(function(d,i){
            return tsFrag.x(i);
        })
        .y(function(d){
            return tsFrag.sortableY(d);
        });
    d3.select("#graph svg.tsv-svg-component").append("g")
        .attr("id", "mini-container")
        .attr("transform", "translate(0, 300)")
        .attr("class", "sortable");

    // Draw all TS lines
    var i = 0;
    var miniLines = d3.select("#mini-container")
        .selectAll("g")
        .data(tsFrag.tsDataArray)
        .enter()                                // Virtual Collection
        .append("g")
        .attr("class", "list-item")
        .attr("transform", function() { return "translate(0, " + (tsFrag.height * i++) + ")";})
        .attr("width", tsFrag.width + tsFrag.smallMargin.left + tsFrag.smallMargin.right)
        .attr("height", tsFrag.height + tsFrag.smallMargin.top + tsFrag.smallMargin.bottom)
        .on("click", selectLineData);

    miniLines.append("rect")
        .attr("width", tsFrag.width + tsFrag.smallMargin.left + tsFrag.smallMargin.right)
        .attr("height", tsFrag.height)
        .attr('fill', function(d) { return ColSch_getAbsoluteGradientColorString(d[tsFrag.relevantColoringFeature]); });

    miniLines = miniLines.append("g")
        .attr("transform", "translate(" + tsFrag.smallMargin.left + "," + tsFrag.smallMargin.top + ")")
        .attr('height', tsFrag.height);

    // Add an overlay rect to listen to mouse events
    miniLines.append("rect")
        .attr("class", "graph-timeSeries-rect overlay")
        .attr("width", tsFrag.width)
        .attr("height", tsFrag.height)
        .attr('fill', "#ffffff");

    // Draw the actual lines and set a scaling factor for each one of them.
    miniLines.append("path")
        .attr("class", "tsv-line")
        .attr("d", function(d) { tsFrag.sortableY = d3.scale.linear().domain([d.min, d.max]).range([tsFrag.height, 0]); return tsFrag.sortableline(d.data); })
        .attr('class', 'tsv-line tsv-colored-line mini')
        .attr("style", function(d) { return "stroke:" + ColSch_getAbsoluteGradientColorString(d[tsFrag.relevantColoringFeature]); });

    miniLines.append("text")
        .attr("class", "tsv-y tsv-label")
        .attr("text-anchor", "end")
        .attr("y", 6)
        .attr("dy", ".75em")
        .attr("transform", "translate(-5,0)")
        .text(function(d) {return d.label;});


    var xAxisForMinis = function() {
        tsFrag.xAxisScale = tsFrag.x;
        tsFrag.xAxisScale.domain()[0] *= tsFrag.samplePeriod;
        tsFrag.xAxisScale.domain()[1] *= tsFrag.samplePeriod;
        return d3.svg.axis().scale(tsFrag.xAxisScale).tickSize(-tsFrag.height * tsFrag.tsDataArray.length).tickSubdivide(true);
    };

    // Add the x-axis.
    d3.select("#mini-container").append("g")
        .attr("class", "tsv-x tsv-axis")
        .attr("width", tsFrag.width)
        .attr("height", tsFrag.height * tsFrag.tsDataArray.length)
        .attr('fill', "#ffffff")
        .attr("transform", "translate(80, " + (tsFrag.height * tsFrag.tsDataArray.length) + ")")
        .call(xAxisForMinis());

    // Make the svg blocks sortable with jQuery
    $(function () {
        var originalPosition, destination;

        function cleanupSVGTemps(trashVisible) {
            d3.selectAll("#ts-trash-can").classed("trash-show", trashVisible);
            d3.selectAll("#ts-trash-can").classed("trash-hidden", !trashVisible);
            destination = null;
            if (!trashVisible) {
                d3.selectAll(".tsv-moving").classed("tsv-moving", false);
            }
        }

        function dropInTrash(ui) {
            // Remove the element dropped on #sortable-delete
            if (tsFrag.tsDataArray.length > 1) {
                var deleteLabel = ui.item[0].__data__.label;
                tsFrag.tsDataArray = tsFrag.tsDataArray.filter(function (obj) {
                    return obj.label !== deleteLabel;
                });
                ui.item.remove();
                tsFrag.selectedIndex = 0;
            }
            drawGraphs();
        }

        function dropInList(destinationPosition) {
            // move element in the main array too.
            move(tsFrag.tsDataArray, originalPosition, destinationPosition);
            // redraw the graph and set the moved element as selected.
            drawGraphs();
            selectLineData("", destinationPosition);
            // Change the sorting selector value to manual
            $("#sortingSelector").val('manual');
        }

        $(".sortable").sortable({
            items: '> g.list-item', // this is used to avoid dragging UI elements.
            cursor: "url(/static/style/img/cursor_move_tsv.png), move",
            connectWith: '#sortable-delete,g#mini-container',
            start: function (e, ui) {
                originalPosition = ui.item.index();
                cleanupSVGTemps(true);
                d3.select("g.list-item:nth-of-type(" + (originalPosition + 1) + ")")
                    .classed("tsv-moving", true);
            },
            stop: function () {
                if (destination != null) {
                    dropInList(destination);
                }
                cleanupSVGTemps(false);
            },
            receive: function (e, ui) {
                if (this.id == 'sortable-delete') {
                    dropInTrash(ui);
                    destination = null;
                }
            }
        });

        $("g.list-item").bind("mouseover", function () {
            destination = $(this).index();
            if (destination > originalPosition) {
                destination = destination - 1;
            }
        });

        $("#tsMoveArea").bind("mouseleave", function () {
            cleanupSVGTemps(false);
            $(".sortable").sortable("cancel");
        });
    });
}

// ====================================    DRAWING FUNCTIONS END   ===========================================
// ====================================    HELPER FUNCTIONS START  ===========================================

/**
 * Get an array containing the time series for the specified [x,y,z] voxel.
 * Out of bound indexes are checked only on the server side.
 * @param x Integer, the x coordinate.
 * @param y Integer, the y coordinate.
 * @param z Integer, the z coordinate.
 * @returns Array of length <code>tsFra.timeLength</code> with the requested time series values
 */
function getPerVoxelTimeSeries(x, y, z){
    x = "x=" + x;
    y = ";y=" + y;
    z = ";z=" + z;
    var query = tsFrag.urlTSData + x + y + z;
    return HLPR_readJSONfromFile(query);
}

/**
 * Compute the variance of an array. The mean can be pre-computed.
 * Based on the same function from science.js at www.github.com/jasondavies/science.js
 * @param arr {Array} The array containing the time series we want to analyze
 * @param mean {number} Optional. The mean of the array.
 * @returns Number - The variance of the array.
 */
function variance(arr, mean){
    var n = arr.length;
    if(n < 1){
        return NaN;
    }
    if(n === 1){
        return 0;
    }
    mean = mean || d3.mean(arr);
    var i = -1;
    var sum = 0;
    while(++i < n){
        var v = arr[i] - mean;
        sum += v * v;
    }
    return sum / (n - 1);
}

/**
 * Compute the covariance of an array.
 * @param tsA {Array} The first array containing the time series we want to analyze
 * @param tsB {Array} The second array containing the time series we want to analyze
 * @param meanA {number} Optional. The mean of the tsA.
 * @param meanB {number} Optional. The mean of the tsB.
 * @returns Number - The covariance between the two arrays.
 */
function covariance(tsA, tsB, meanA, meanB){
    var n = tsA.length;
    if(n < 1){
        return NaN;
    }
    if(n === 1){
        return 0;
    }
    var i = -1;
    var sum = 0;
    meanA = meanA ||d3.mean(tsA);
    meanB = meanB || d3.mean(tsB);
    while(++i < n){
        var diffA = tsA[i] - meanA;
        var diffB = tsB[i] - meanB;
        sum += diffA * diffB;
    }
    return sum / n;
}

/**
 * Select a graph line and mark it as highlighted
 * @param d Data. We need it here because d3 always passes data as the first parameter.
 * @param i The clicked line index. D3 always passes this variable as second parameter.
 */
function selectLineData(d, i){
    tsFrag.selectedIndex = i;

    //remove the highlight classes
    d3.selectAll(".tsv-highlight")
        .classed("tsv-highlight", false);
    d3.selectAll(".tsv-text-highlight")
        .classed("tsv-text-highlight", false);

    //add the highlight class
    d3.select("path.tsv-colored-line:nth-of-type(" + (i + 1) +")")
        .classed("tsv-highlight", true);
    d3.select("path.tsv-colored-line.mini:nth-of-type(" + (i + 1) +")")
        .classed("tsv-highlight", true);
    d3.select("#graph g.list-item:nth-of-type(" + (i + 1) +") text")
        .classed("tsv-text-highlight", true);
}

/**
 *  Attach event listeners to all UI selectors
 */
function attachUIListeners(){
    // Attach sorting listener to Sorting Selector
    $("#sortingSelector").change(function(e){
        var pvt = {x: tsFrag.selectedEntity[0], y: tsFrag.selectedEntity[1], z: tsFrag.selectedEntity[2]};
        sortTsGraphs(e.currentTarget.value, tsFrag.relevantSortingFeature, pvt);
        // redraw the graph
        drawGraphs();
    });

    // Attach sorting listener to Relevant Feature Selector
    $("#relevantFeatureSelector").change(function(e){
        tsFrag.relevantSortingFeature = e.currentTarget.value;
        var pvt = {x: tsFrag.selectedEntity[0], y: tsFrag.selectedEntity[1], z: tsFrag.selectedEntity[2]};
        sortTsGraphs($("#sortingSelector").val(), tsFrag.relevantSortingFeature, pvt);
        // redraw the graph
        drawGraphs();
    });

    // Attach sorting listener to Relevant Color Selector
    $("#colorBySelector").change(function(e){
        tsFrag.relevantColoringFeature = e.currentTarget.value;
        // redraw the graph
        drawGraphs();
    });

    // Set the proper trash bin dimensions
    var trashElem = $("#ts-trash-can");
    var sortableElem = $("#sortable-delete");
    trashElem.height(sortableElem.height() + 17);
    trashElem.width(sortableElem.width() + 17);
}

/**
 * Callback function for the on-mouse-move event
 */
function TSF_mousemove() {
    var xPos = d3.mouse(this)[0];
    // Reset X Domain, to get a correct reading:
    tsFrag.x.domain([0, tsFrag.timeLength]);
    var arrayIdx = Math.floor(tsFrag.x.invert(xPos));
    var currentValue = tsFrag.tsDataArray[tsFrag.selectedIndex].data[arrayIdx];
    var selectedLine = d3.select("path.tsv-colored-line:nth-of-type(" + (tsFrag.selectedIndex + 1) + ")");

    var mouseOverTooltipValue = d3.select(".focus");
    mouseOverTooltipValue.attr("transform", "translate(" + xPos + "," + tsFrag.y(currentValue) + ")");
    mouseOverTooltipValue.select("text").text(currentValue);

    //Move blue line following the mouse
    var pathLength = selectedLine.node().getTotalLength();
    // the +-3 lets us click the graph and not the line
    xPos = xPos > ( pathLength / 2 ) ? Math.min(xPos + 3, pathLength) : Math.max(xPos - 3, 0);
    d3.select(".verticalLine").attr("transform", function () {
        return "translate(" + xPos + ",0)";
    });

    var X = xPos;
    var beginning = X,
        end = pathLength,
        target;
    var pos;

    while (true) {
        target = Math.floor((beginning + end) / 2);
        pos = selectedLine.node().getPointAtLength(target);
        if ((target === end || target === beginning) && pos.x !== X) {
            break;
        }
        if (pos.x > X) {
            end = target;
        } else if (pos.x < X) {
            beginning = target;
        } else {
            break; //position found
        }
    }

    var mouseCircle = d3.select("#mouseLineCircle");
    mouseCircle.attr("opacity", 1)
        .attr("cx", X)
        .attr("cy", pos.y);

    mouseOverTooltipValue.attr("transform", "translate(" + X + "," + pos.y + ")");
}

/**
* Moves element in arr[old_index] to arr[new_index]. It allows for negative indexes so
* moving he last element to the nth position can be written as <code>move(arr, -1, n)</code>
* while moving the nth element to the last position can be <code>move(arr, n, -1)</code>
*
* @param arr The array to be modified
* @param old_index The index of the element to be moved
* @param new_index The index where to move the element
*/
function move(arr, old_index, new_index){
    while(old_index < 0){
        old_index += arr.length;
    }
    while(new_index < 0){
        new_index += arr.length;
    }
    if(new_index >= arr.length){
        var k = new_index - arr.length;
        while((k--) + 1){
            arr.push(undefined);
        }
    }
    arr.splice(new_index, 0, arr.splice(old_index, 1)[0]);
}

/**
 * Sort the time-series array using different sorting functions
 * @param order    String, can be desc, asc, xyz. The sorting order we want to use.
 * @param by       String, name of the attribute we should consider when sorting.
 * @param pivot    Object with x,y,z coordinates. Used only by "xyz" sort.
 */
function sortTsGraphs(order, by, pivot){
    // manhattan distance helper for xyz sorting.
    function md3d(a, b){
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
    }
    // sorting from biggest to smallest
    if (order === "descending"){
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : + (a[by] < b[by]) || -1;
        });
    }
    // sorting from smallest to biggest
    else if (order === "ascending") {
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : + (a[by] > b[by]) || -1;
        });
    }
    // sorting based on manhattan distance from the pivot coordinate
    else if (order === "manhattan") {
        pivot = pivot || {x:0,y:0,z:0};
        tsFrag.tsDataArray.sort(function(a, b){
            a = md3d(a, pivot);
            b = md3d(b, pivot);
          return a == b ? 0 : +(a > b) || -1;
        });
    }
}
/**
 * Update the brushes based on the current time point
 */
function updateBrush() {
    var bMin = Math.max(0,tsFrag.currentTimePoint-30);
    var bMax = Math.min(tsFrag.currentTimePoint+30,tsFrag.timeLength);
    d3.select('.tsv-brush').transition()
      .delay(0)
      .call(tsFrag.brush.extent([bMin, bMax]))
      .call(tsFrag.brush.event);
}

function TSF_updateTimeGauge(timePoint){
    d3.select(".timeVerticalLine").attr("transform", function(){
                var width = $(".graph-timeSeries-rect").attr("width");
                var pos = (timePoint * width) / (tsFrag.timeLength);
                return "translate(" + pos + ", 0)";
            });
}

// ====================================    HELPER FUNCTIONS END    ===========================================

// MODULE EXPORTS
window.updateTSFragment = updateTSFragment;
window.TSF_initVisualizer = TSF_initVisualizer;
window.drawGraphs = drawGraphs;
window.TSF_updateTimeGauge = TSF_updateTimeGauge;
// debugging purposes only export
window._debug_tsFrag = tsFrag;
})();