var tsFrag = {
    dataTimeSeries: "",                 // Contains the address to query the time series of a specific voxel.
    m: [],                              // Global preview margin sizes
    w: 0,                               // Global preview width
    h: 0,                               // Global preview width
    smallMargin: {},                    // Sortable time series margins
    width: 0,                           // Sortable time series width
    height: 0,                          // Sortable time series height
    tsDataArray: [],                    // Array containing all the time series data
    selectedInex: 0,                    // Selected time series line index
    relevantFeature: "mean",            // Stores what feature do we care about when sorting the time series.
    timeLength: 0,
    samplePeriod: 0,
    samplePeriodUnit: "",
};

function TSF_initVisualizer(){
    tsFrag.timeLength = tsVol.timeLength;
    tsFrag.samplePeriod = tsVol.samplePeriod;
    tsFrag.samplePeriodUnit = tsVol.samplePeriodUnit;
    tsFrag.dataTimeSeries = tsVol.dataTimeSeries;
    tsFrag.minimumValue = tsVol.minimumValue;
    tsFrag.maximumValue = tsVol.maximumValue;
}

// ==================================== PER VOXEL TIMESERIES START ===========================================

function getPerVoxelTimeSeries(x,y,z){
    x = "x=" + x;
    y = ";y=" + y;
    z = ";z=" + z;
    var query = tsVol.dataTimeSeries + x + y + z;
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
    this.x = params.x || tsVol.selectedEntity[0],
    this.y =  params.y || tsVol.selectedEntity[1],
    this.z = params.z || tsVol.selectedEntity[2],
    this.label = params.label || "["+this.x+","+this.y+","+this.z+"]",
    this.data = params.data || data,
    this.max = params.max || d3.max(data),
    this.min = params.min || d3.min(data),
    this.mean = params.mean || d3.mean(data),
    this.median = params.median || d3.median(data),
    this.variance = params.variance || variance(data, this.mean),
    this.deviation = params.deviation || Math.sqrt(this.variance)
}

/* implementation heavily influenced by http://bl.ocks.org/1166403 */
function drawGraphs(){
    var selected = 0;
    $('#graph').empty();
    // define dimensions of graph
    var m = [30, 80, 30, 80]; // margins
    var smallMargin = {top: 0, right: 80, bottom: 0, left: 80};
    var width = $('#graph').width() - smallMargin.left - smallMargin.right;
    var height = 30 - smallMargin.top - smallMargin.bottom;
    var w = $('#graph').width() - m[1] - m[3]; // width
    var h = 240 - m[0] - m[2]; // height

    var label = "["+tsVol.selectedEntity[0]+","+tsVol.selectedEntity[1]+","+tsVol.selectedEntity[2]+"]";
    if(!containsByLabel(tsFrag.tsDataArray, label)){
        var tmp = new tsDataObj({}, getPerVoxelTimeSeries(tsVol.selectedEntity[0], tsVol.selectedEntity[1], tsVol.selectedEntity[2]));
        tsFrag.tsDataArray.push(tmp);
        var pvt = {x: tsVol.selectedEntity[0], y:  tsVol.selectedEntity[1],z:  tsVol.selectedEntity[2]};
        sortTsGraphs($("#sortingSelector").val(), tsFrag.relevantFeature, pvt);
    }
    if(tsFrag.tsDataArray.length < 1){
        return;
    }

    // X scale will fit all values from data[] within pixels 0-w
    var x = d3.scale.linear().domain([0, tsFrag.timeLength]).range([0, w]);

    var localMax = d3.max(tsFrag.tsDataArray, function(array) {
      return array.max;
    });
    var localMin = d3.min(tsFrag.tsDataArray, function(array) {
      return array.min;
    });
    var y = d3.scale.linear().domain([localMin, localMax]).range([h, 0]);

    // create a line function that can convert data[] into x and y points
    var line = d3.svg.line()
        //.interpolate("basis") //basis for spline interpolation
        // assign the X function to plot our line as we wish
        .x(function(d,i) { 
            // return the X coordinate where we want to plot this datapoint
            return x(i); 
        })
        .y(function(d) { 
            // return the Y coordinate where we want to plot this datapoint
            return y(d); 
        });

    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select("#graph").append("svg:svg")
          .attr("width", w + m[1] + m[3])
          .attr("height", h + m[0] + m[2])
          .attr("class", "graph-svg-component")
        .append("svg:g")
          .attr("transform", "translate(" + m[3] + "," + m[0] + ")");

    var rect = graph.append("rect")
        .attr('w',0)
        .attr('h',0)
        .attr('width',w)
        .attr('height',h)
        .attr('fill', "#ffffff")
        .attr("class", "graph-timeSeries-rect")

    graph.append("rect")
        .attr("class", "overlay")
        .attr("width", w)
        .attr("height", h)
        .attr('fill', "#ffffff")
        .on("mouseover", function() { focus.style("display", null); })
        .on("mouseout", function() { focus.style("display", "none"); })
        .on("mousemove", mousemove);

    // create xAxis
    var xAxixScale = d3.scale.linear().domain([0, tsFrag.timeLength*tsFrag.samplePeriod]).range([0, w]);
    var xAxis = d3.svg.axis().scale(xAxixScale).tickSize(-h).tickSubdivide(true);
    // Add the x-axis.
    graph.append("svg:g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + h + ")")
        .call(xAxis);

    var timeUnit = tsFrag.samplePeriodUnit=="sec" ? "Seconds" : tsFrag.samplePeriodUnit;
    graph.append("text")
        .attr("class", "x axis")
        .attr("text-anchor", "end")
        .attr("x", w)
        .attr("y", h -8 )
        .text("Time in " + timeUnit);

    // create left yAxis
    var yAxisLeft = d3.svg.axis().scale(y).ticks(4).orient("left");
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
        .attr("y", "-1em")
        .attr("dy", ".75em")
        //.attr("transform", "rotate(-90)")
        .text("Measurements");

    graph.selectAll('.line')
        .data(tsFrag.tsDataArray)
        .enter()
        .append("path")
            .attr("class", "line")
            //.attr("clip-path", "url(#clip)")
            .attr("d", function(d){return line(d.data);})
            .attr('class', 'line colored-line')
            .attr("style", function(d){return "stroke:" + getGradientColorString(d.mean, tsFrag.minimumValue, tsFrag.maximumValue);} )
            .on("mouseover", selectLineData);

    // Add an SVG element for each symbol, with the desired dimensions and margin.
    d3.select("#graph").append("ul")
        .attr("id", "mini-container")
        .attr("class", "sortable");
    var svg = d3.select("#mini-container").selectAll("svg")
        .data(tsFrag.tsDataArray)
    .enter()
    .append("li")
    .append("svg")
        .attr("width", width + smallMargin.left + smallMargin.right)
        .attr("height", height + smallMargin.top + smallMargin.bottom)
        .attr("class", "graph-svg-component")
        .attr("style", function(d){return "background-color:" + getGradientColorString(d.mean, tsFrag.minimumValue, tsFrag.maximumValue);} )
        .attr("display", "block")
        .on("click", selectLineData)
    .append("g")
        .attr("transform", "translate(" + smallMargin.left + "," + smallMargin.top + ")")
        .attr('height', height)

    svg.append("rect")
        .attr("class", "graph-timeSeries-rect overlay")
        .attr("width", width)
        .attr("height", height)
        .attr('fill', "#ffffff")
        .on("mouseover", function() { focus.style("display", null); })
        .on("mouseout", function() { focus.style("display", "none"); })
        .on("mousemove", mousemove);

    svg.append("path")
        .attr("class", "line")
        .attr("d", function(d) { y = d3.scale.linear().domain([d.min, d.max]).range([height, 0]); return line(d.data); })
        .attr('class', 'line colored-line mini')
        .attr("style", function(d){return "stroke:" + getGradientColorString(d.mean, tsFrag.minimumValue, tsFrag.maximumValue);} )

    svg.append("text")
        .attr("class", "y label")
        .attr("text-anchor", "end")
        .attr("y", 6)
        .attr("dy", ".75em")
        .attr("transform", "translate(-5,0)")
        .text(function(d){return d.label;});

    var focus = graph.append("g")
        .attr("class", "focus")
        .style("display", "none");

    focus.append("circle")
        .attr("r", 4.5);

    focus.append("text")
        .attr("x", 9)
        .attr("dy", ".35em")
        .attr("style","background-color: aliceblue");

    function mousemove() {
        var x0 = x.invert(d3.mouse(this)[0]),
            i = Math.floor(x0),
            data = tsFrag.tsDataArray[selected].data,
            d0 = data[i - 1 ],
            d1 = data[i],
            d = x0 - d0 > d1 - x0 ? d1 : d0;

        var selectedLine = d3.select("path.colored-line:nth-of-type(" + (selected+1) +")");

        focus.attr("transform", "translate(" + d3.mouse(this)[0] + "," + y(data[i]) + ")");
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
        circle.attr("opacity", 1)
            .attr("cx", X)
            .attr("cy", pos.y);

        focus.attr("transform", "translate(" + X + "," + pos.y + ")");
    }

    function selectLineData(d, i) {
        //We need to include "d", since the index will
        //always be the second value passed in to the function
        selected = i;
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

    var verticalLine = graph.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', h)
        .attr("stroke", "steelblue")
        .attr('class', 'verticalLine');

    var timeVerticalLine = graph.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', h)
        .attr("stroke", "red")
        .attr('class', 'timeVerticalLine');

    var circle = graph.append("circle")
        .attr("opacity", 0)
        .attr('r', 2)
        //.attr('fill', 'darkred');

    $("#time-position").on("slide change", function(){
        //Move red line following the mouse
        var wdt = $(".graph-timeSeries-rect").attr("width");
        var xPos = (tsVol.currentTimePoint*wdt)/(tsFrag.timeLength);

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

    /*
    * Moves element in arr[old_index] to arr[new_index]
    * @param arr The array to be modified
    * @param old_index The index of the element to be moved
    * @param new_index The index where to move the element
    * @returns Nothig, it modifies arr directly
    */
    function move(arr, old_index, new_index) {
        while (old_index < 0) {
            old_index += arr.length;
        }
        while (new_index < 0) {
            new_index += arr.length;
        }
        if (new_index >= arr.length) {
            var k = new_index - arr.length;
            while ((k--) + 1) {
                arr.push(undefined);
            }
        }
        arr.splice(new_index, 0, arr.splice(old_index, 1)[0]);
    };


    /*
        This is what allow us to manually sort the svg blocks.
        The draging is smot because of the <li> tags on HTML, do not remove.
        TODO: Make the manual sorting consistent with the array
        structure. (easy)
    */
    $(function() {
        var originalPosition, destination;
        $( ".sortable" ).sortable({
            items: '> li:not(.pin)', // this is used to avoid dragging UI elementstime
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
                asd = ui;
                if(this.id == 'sortable-delete'){
                    // Remove the element dropped on #sortable-delete
                    if(tsFrag.tsDataArray.length > 1){
                        var deleteLabel = ui.item[0].__data__.label;
                        tsFrag.tsDataArray = tsFrag.tsDataArray.filter(function(obj){
                            return obj.label != deleteLabel;
                        })
                        ui.item.remove();
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

    $("#sortable-delete").width()
    if($("#mini-container").children().length < 2){
        $("#mini-container").sortable( "disable" );
    }
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
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y)+Math.abs(a.z - b.z);
    } 
    // sorting from biggest to smallest
    if(order == "descending"){
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : +(a[by] < b[by]) || -1;
        });
    }
    // sorting from smallest to biggest
    else if(order == "ascending"){
        tsFrag.tsDataArray.sort(function(a, b){
          return a[by] == b[by] ? 0 : +(a[by] > b[by]) || -1;
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
        return
    }
}

// Attach sorting listener to sorting selectors
$(function(){
    $("#sortingSelector").change(function(e){
        var pvt = {x: tsVol.selectedEntity[0], y: tsVol.selectedEntity[1], z: tsVol.selectedEntity[2]};
        sortTsGraphs(e.currentTarget.value, tsFrag.relevantFeature, pvt);
        // redraw the graph
        drawGraphs();
    });
});

// Attach sorting listener to sorting selector
$(function(){
    $("#relevantFeatureSelector").change(function(e){
        tsFrag.relevantFeature = e.currentTarget.value;
        var pvt = {x: tsVol.selectedEntity[0], y: tsVol.selectedEntity[1], z: tsVol.selectedEntity[2]};
        sortTsGraphs($("#sortingSelector").val(), tsFrag.relevantFeature, pvt);
        // redraw the graph
        drawGraphs();
    });
});

$(function(){
    $("#ts-trash-can").height($("#sortable-delete").height()+17);
    $("#ts-trash-can").width($("#sortable-delete").width()+17);
})



// ==================================== PER VOXEL TIMESERIES END =============================================