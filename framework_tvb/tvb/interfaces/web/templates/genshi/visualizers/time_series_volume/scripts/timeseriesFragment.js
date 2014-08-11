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
    timeLength: 0,
    samplePeriod: 0,
    samplePeriodUnit: "",
    selectedEntity: [],
    line: null,
    x: null,
    y: null
};

function TSF_initVisualizer(tsDataRequest){
    tsFrag.dataAddress = tsDataRequest;

    // take all the necessary data from the time series volume visualizer
    tsFrag.timeLength = tsVol.timeLength;
    tsFrag.samplePeriod = tsVol.samplePeriod;
    tsFrag.samplePeriodUnit = tsVol.samplePeriodUnit;
    tsFrag.dataTimeSeries = tsVol.dataTimeSeries;
    tsFrag.minimumValue = tsVol.minimumValue;
    tsFrag.maximumValue = tsVol.maximumValue;

    // tsFrag own data
    // define dimensions of graph
    tsFrag.m = [30, 80, 30, 80]; // margins
    tsFrag.smallMargin = {top: 0, right: 80, bottom: 0, left: 80};
    tsFrag.width = $('#graph').width() - tsFrag.smallMargin.left - tsFrag.smallMargin.right;
    tsFrag.height = 30 - tsFrag.smallMargin.top - tsFrag.smallMargin.bottom;
    tsFrag.w = $('#graph').width() - tsFrag.m[1] - tsFrag.m[3]; // width
    tsFrag.h = 240 - tsFrag.m[0] - tsFrag.m[2]; // height
}

function updateTSFragment(){
    tsFrag.selectedEntity = tsVol.selectedEntity;
    tsFrag.currentTimePoint = tsVol.currentTimePoint;
}

// ==================================== PER VOXEL TIMESERIES START ===========================================

function getPerVoxelTimeSeries(x,y,z){
    x = "x=" + x;
    y = ";y=" + y;
    z = ";z=" + z;
    var query = tsFrag.dataAddress + x + y + z;
    return HLPR_readJSONfromFile(query);;
}


//Check if an array of objects contains another object with a given 'label' attribute.
function containsByLabel(a, label) {
    for (var i = 0; i < a.length; i++) {
        if (a[i].label === label) {
            return true;
        }
    }
    return false;
}

function variance(arr, mean) {
    var n = arr.length;
    if(n < 1){
        return NaN;
    }
    if(n === 1){
        return 0;
    }
    mean = mean || d3.mean(arr);
    var i = -1;
    var s = 0;
    while (++i < n){
        var v = arr[i] - mean;
        s += v * v;
    }
    return s / (n - 1);
};

function covariance(tsA, tsB){
    var sum = 0;
    var mA = d3.mean(tsA);
    var mB = d3.mean(tsB);
    for( var k in tsA){
        var diffA = tsA[k] - mA;
        var diffB = tsB[k] - mB;
        sum += diffA * diffB;
    }
    return sum / tsA.length;
}

var tsDataObj = function(params, data){
    this.x = params.x || tsFrag.selectedEntity[0],
    this.y =  params.y || tsFrag.selectedEntity[1],
    this.z = params.z || tsFrag.selectedEntity[2],
    this.label = params.label || "["+this.x+","+this.y+","+this.z+"]",
    this.data = params.data || data,
    this.max = params.max || d3.max(data),
    this.min = params.min || d3.min(data),
    this.mean = params.mean || d3.mean(data),
    this.median = params.median || d3.median(data),
    this.variance = params.variance || variance(data, this.mean),
    this.deviation = params.deviation || Math.sqrt(this.variance)
}

function selectLineData(d, i) {
    tsFrag.selectedIndex = i;
    //remove the highlight class
    d3.selectAll(".highlight")
        .classed("highlight", false);
    d3.selectAll(".text-highlight")
        .classed("text-highlight", false);

    //add the highlight class
    d3.select("path.colored-line:nth-of-type(" + (i+1) +")")
        .classed("highlight", true);
    d3.select("#graph li:nth-of-type(" + (i+1) +") text")
        .classed("text-highlight", true);
}

function mousemove() {
    var x0 = tsFrag.x.invert(d3.mouse(this)[0]),
        i = Math.floor(x0),
        data = tsFrag.tsDataArray[tsFrag.selectedIndex].data,
        d0 = data[i - 1 ],
        d1 = data[i],
        d = x0 - d0 > d1 - x0 ? d1 : d0;
    var selectedLine = d3.select("path.colored-line:nth-of-type(" + (tsFrag.selectedIndex+1) +")");

    var focus = d3.select(".focus");

    focus.attr("transform", "translate(" + d3.mouse(this)[0] + "," + tsFrag.y(data[i]) + ")");
    focus.select("text").text(d1);

    //Move blue line following the mouse
    var xPos = d3.mouse(this)[0];
    // the +-3 lets us click the graph and not the line
    var pathLength = selectedLine.node().getTotalLength();

    xPos = xPos > ( pathLength / 2 ) ? Math.min(xPos+3, pathLength) : Math.max(xPos-3, 0);
    d3.select(".verticalLine").attr("transform", function(){
        return "translate(" + xPos + ",0)";
    });

    var X = xPos;
    var beginning = X,
        end = pathLength,
        target;
    while (true) {
        target = Math.floor((beginning + end) / 2);
        pos = selectedLine.node().getPointAtLength(target);
        if ((target === end || target === beginning) && pos.x !== X) {
            break;
        }
        if (pos.x > X) end = target;
        else if (pos.x < X) beginning = target;
        else break; //position found
    }
    var circle = d3.select("#mouseLineCircle");
    circle.attr("opacity", 1)
        .attr("cx", X)
        .attr("cy", pos.y);

    focus.attr("transform", "translate(" + X + "," + pos.y + ")");
}

function drawGobalTimeseries(){
    tsFrag.x = d3.scale.linear().domain([0, tsFrag.timeLength]).range([0, tsFrag.w]);

    var localMax = d3.max(tsFrag.tsDataArray, function(array) {
      return array.max;
    });
    var localMin = d3.min(tsFrag.tsDataArray, function(array) {
      return array.min;
    });
    tsFrag.y = d3.scale.linear().domain([localMin, localMax]).range([tsFrag.h, 0]);

    // create a line function that can convert data[] into x and y points
    tsFrag.line = d3.svg.line()
        //.interpolate("basis") //basis for spline interpolation
        // assign the X function to plot our line as we wish
        .x(function(d,i) {
            // return the X coordinate where we want to plot this datapoint
            return tsFrag.x(i);
        })
        .y(function(d) {
            // return the Y coordinate where we want to plot this datapoint
            return tsFrag.y(d);
        });

    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select("#graph").append("svg:svg")
          .attr("width", tsFrag.w + tsFrag.m[1] + tsFrag.m[3])
          .attr("height", tsFrag.h + tsFrag.m[0] + tsFrag.m[2])
          .attr("class", "graph-svg-component")
        .append("svg:g")
          .attr("transform", "translate(" + tsFrag.m[3] + "," + tsFrag.m[0] + ")");

    var rect = graph.append("rect")
        .attr('w',0)
        .attr('h',0)
        .attr('width', tsFrag.w)
        .attr('height', tsFrag.h)
        .attr('fill', "#ffffff")
        .attr("class", "graph-timeSeries-rect")

    graph.append("rect")
        .attr("class", "overlay")
        .attr("width", tsFrag.w)
        .attr("height", tsFrag.h)
        .attr('fill', "#ffffff")
        .on("mouseover", function() { focus.style("display", null); })
        .on("mouseout", function() { focus.style("display", "none"); })
        .on("mousemove", mousemove);

    // create xAxis
    var xAxixScale = d3.scale.linear().domain([0, tsFrag.timeLength*tsFrag.samplePeriod]).range([0, tsFrag.w]);
    var xAxis = d3.svg.axis().scale(xAxixScale).tickSize(-tsFrag.h).tickSubdivide(true);
    // Add the x-axis.
    graph.append("svg:g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + tsFrag.h + ")")
        .call(xAxis);

    var timeUnit = tsFrag.samplePeriodUnit=="sec" ? "Seconds" : tsFrag.samplePeriodUnit;
    graph.append("text")
        .attr("class", "x axis")
        .attr("text-anchor", "end")
        .attr("x", tsFrag.w)
        .attr("y", tsFrag.h - 8 )
        .text("Time in " + timeUnit);

    // create left yAxis
    var yAxisLeft = d3.svg.axis().scale(tsFrag.y).ticks(4).orient("left");
    // Add the y-axis to the left
    graph.append("svg:g")
          .attr("class", "y axis")
          .attr("transform", "translate(-25,0)")
          .call(yAxisLeft);
    //
    graph.append("text")
        .attr("class", "y axis")
        .attr("text-anchor", "end")
        .attr("x", "1em")
        .attr("y", "-1.5em")
        .attr("dy", ".75em")
        //.attr("transform", "rotate(-90)")
        .text("Measurements");

    graph.selectAll('.line')
        .data(tsFrag.tsDataArray)
        .enter()
        .append("path")
            .attr("class", "line")
            //.attr("clip-path", "url(#clip)")
            .attr("d", function(d){return tsFrag.line(d.data);})
            .attr('class', 'line colored-line')
            .attr("style", function(d){return "stroke:" + getGradientColorString(d[tsFrag.relevantColoringFeature], tsFrag.minimumValue, tsFrag.maximumValue);} )
            .on("mouseover", selectLineData);
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

    var timeVerticalLine = graph.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', tsFrag.h)
        .attr("stroke", "red")
        .attr('class', 'timeVerticalLine');

    var circle = graph.append("circle")
        .attr("opacity", 0)
        .attr('r', 2)
        .attr("id", "mouseLineCircle");
        //.attr('fill', 'darkred');

    $("#time-position").on("slide change", function(){
        //Move red line following the mouse
        var wdt = $(".graph-timeSeries-rect").attr("width");
        var xPos = (tsFrag.currentTimePoint*wdt)/(tsFrag.timeLength);

        var selectedLine = d3.select("path.highlight");
        if(selectedLine[0][0] == null){
            selectedLine = d3.select("path.colored-line:nth-of-type(1)")
        }
        var pathLength = selectedLine.node().getTotalLength();
        var X = xPos;
        var beginning = X,
            end = pathLength,
            target;
        while (true) {
            target = Math.floor((beginning + end) / 2);
            pos = selectedLine.node().getPointAtLength(target);
            if ((target === end || target === beginning) && pos.x !== X) {
                break;
            }
            if (pos.x > X) end = target;
            else if (pos.x < X) beginning = target;
            else break; //position found
        }
    });
}

function drawSortableGraph(){
    d3.select("#graph").append("ul")
        .attr("id", "mini-container")
        .attr("class", "sortable");
    var svg = d3.select("#mini-container").selectAll("svg")
        .data(tsFrag.tsDataArray)
    .enter()
    .append("li")
    .append("svg")
        .attr("width", tsFrag.width + tsFrag.smallMargin.left + tsFrag.smallMargin.right)
        .attr("height", tsFrag.height + tsFrag.smallMargin.top + tsFrag.smallMargin.bottom)
        .attr("class", "graph-svg-component")
        .attr("style", function(d){return "background-color:" + getGradientColorString(d[tsFrag.relevantColoringFeature], tsFrag.minimumValue, tsFrag.maximumValue);} )
        .attr("display", "block")
        .on("click", selectLineData)
    .append("g")
        .attr("transform", "translate(" + tsFrag.smallMargin.left + "," + tsFrag.smallMargin.top + ")")
        .attr('height', tsFrag.height)

    svg.append("rect")
        .attr("class", "graph-timeSeries-rect overlay")
        .attr("width", tsFrag.width)
        .attr("height", tsFrag.height)
        .attr('fill', "#ffffff")
        .on("mouseover", function() { d3.select(".focus").style("display", null); })
        .on("mouseout", function() { d3.select(".focus").style("display", "none"); })
        .on("mousemove", mousemove);

    svg.append("path")
        .attr("class", "line")
        .attr("d", function(d) { tsFrag.y = d3.scale.linear().domain([d.min, d.max]).range([tsFrag.height, 0]); return tsFrag.line(d.data); })
        .attr('class', 'line colored-line mini')
        .attr("style", function(d){return "stroke:" + getGradientColorString(d[tsFrag.relevantColoringFeature], tsFrag.minimumValue, tsFrag.maximumValue);} )

    svg.append("text")
        .attr("class", "y label")
        .attr("text-anchor", "end")
        .attr("y", 6)
        .attr("dy", ".75em")
        .attr("transform", "translate(-5,0)")
        .text(function(d){return d.label;});
    /*
        Make the svg blocks sortable
        The draging is smoot because of the <li> tags on HTML, do not remove them.
    */
    $(function() {
        var originalPosition, destination;
        $( ".sortable" ).sortable({
            items: '> li:not(.pin)', // this is used to avoid dragging UI elements.
            cursor: "move",
            connectWith: '#sortable-delete',
            axis: "y",
            revert: 250,
            start: function(event,ui){
                originalPosition = ui.item.index();
                d3.selectAll("#ts-trash-can")
                    .classed("trash-hidden", false);
                d3.selectAll("#ts-trash-can")
                    .classed("trash-show", true);
            },
            stop: function(event,ui){
                d3.selectAll("#ts-trash-can")
                    .classed("trash-show", false);
                d3.selectAll("#ts-trash-can")
                    .classed("trash-hidden", true);
            },
            update: function(event, ui) {
                if(this.id == 'sortable-delete'){
                    // Remove the element dropped on #sortable-delete
                    if(tsFrag.tsDataArray.length > 1){
                        var deleteLabel = ui.item[0].__data__.label;
                        tsFrag.tsDataArray = tsFrag.tsDataArray.filter(function(obj){
                            return obj.label != deleteLabel;
                        })
                        ui.item.remove();
                        tsFrag.selectedIndex = 0;
                    }
                    drawGraphs();
                }else{
                    // move element in the main array too.
                    destination = ui.item.index();
                    move(tsFrag.tsDataArray, originalPosition, destination);
                    // redraw the graph and set the moved element as selected.
                    drawGraphs();
                    selectLineData("", destination);
                    // Change the sorting selector value to manual
                    $("#sortingSelector").val('manual');
                }
            }
        });
        $("#mini-container").disableSelection();
      });
}

/* implementation heavily influenced by http://bl.ocks.org/1166403 */
function drawGraphs(){
    //var selected = 0;
    $('#graph').empty();

    var label = "["+tsFrag.selectedEntity[0]+","+tsFrag.selectedEntity[1]+","+tsFrag.selectedEntity[2]+"]";
    if(!containsByLabel(tsFrag.tsDataArray, label)){
        var tmp = new tsDataObj({}, getPerVoxelTimeSeries(tsFrag.selectedEntity[0], tsFrag.selectedEntity[1], tsFrag.selectedEntity[2]));
        tsFrag.tsDataArray.push(tmp);
        var pvt = {x: tsFrag.selectedEntity[0], y:  tsFrag.selectedEntity[1],z:  tsFrag.selectedEntity[2]};
        sortTsGraphs($("#sortingSelector").val(), tsFrag.relevantSortingFeature, pvt);
    }
    if(tsFrag.tsDataArray.length < 1){
        return;
    }
    drawGobalTimeseries();
    drawSortableGraph();

    if($("#mini-container").children().length < 2){
        $("#mini-container").sortable( "disable" );
        d3.selectAll("#mini-container li")
        .classed("pin", true);
    }
}

/*
* Moves element in arr[old_index] to arr[new_index]. It allows for negative indexes so
* moving he last element to the nth position can be written as <code>move(arr, -1, n)</code>
* while moving the nth element to the last position can be <code>move(arr, n, -1)</code>
* @param arr The array to be modified
* @param old_index The index of the element to be moved
* @param new_index The index where to move the element
* @returns Nothig, it modifies arr directly
*/
function move(arr, old_index, new_index){
    while (old_index < 0){
        old_index += arr.length;
    }
    while (new_index < 0){
        new_index += arr.length;
    }
    if (new_index >= arr.length){
        var k = new_index - arr.length;
        while ((k--) + 1){
            arr.push(undefined);
        }
    }
    arr.splice(new_index, 0, arr.splice(old_index, 1)[0]);
};

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
    if(order == "descending"){
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : + (a[by] < b[by]) || -1;
        });
    }
    // sorting from smallest to biggest
    else if(order == "ascending"){
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : + (a[by] > b[by]) || -1;
        });
    }
    // sorting based on manhattan distance from the pivot coordinate
    else if(order == "manhattan"){
        pivot = pivot || {x:0,y:0,z:0};
        tsFrag.tsDataArray.sort(function(a, b){
            a = md3d(a, pivot);
            b = md3d(b, pivot);
          return a == b ? 0 : +(a > b) || -1;
        });
    }
    else {
        return;
    }
}

$(function(){
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

    $("#ts-trash-can").height($("#sortable-delete").height()+17);
    $("#ts-trash-can").width($("#sortable-delete").width()+17);
});



// ==================================== PER VOXEL TIMESERIES END =============================================