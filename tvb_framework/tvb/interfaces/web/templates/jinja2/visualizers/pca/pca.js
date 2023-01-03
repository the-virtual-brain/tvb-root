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


/*
 * REQUIRES:
 * 		- /static/js/d3.v3.js : for d3 specific functions used for visualization
 * 		- /static/js/tvbviz.js : for tv.util.ndar and operations on those kind of data
 */
PCAViewer = function (root, fractionsDataUrl, weightsDataUrl, labelsData, width, height) {
	/*
	 * Specific visualizer for a PCA result, using d3.js for SVG visualization. Has
	 * two graphs, a pie chart and a data plot.
	 */
	// ------------------------ Global variables -----------------------
	this.parentDivID = "#" + root; 	// Root div to which the SVG visualization is added 
	this.width = width || 800;	// Width of SVG viewer
	this.height = height || 400;	// Height of SVG viewer
	this.fromChan = 0;	// We display only a range of channels, this is start channel
	this.toChan = 10;	// We display only a range of channels, this is end channel
	this.nComp = 0;		// Number of components
	this.colorMap = d3.scale.category10();	// Generate color set for each node
	
	// ------------------ Left side Pie char variables --------------------
	this.fractions = [];	// The fractions data array, 
	this.fractionsDataUrl = fractionsDataUrl;	// url from which to load fractions data
	this.pieInnerRadius = 50;	// inner radius of left side PIE chart
	this.pieOuterRadius = 150;	// outer radius of left side PIE chart
	
	// ------------------ Right side vector plot variables ---------------
	this.weightsRaw = [];	// Weights data array
	this.weightsDataUrl = weightsDataUrl;	// url from which to load weight data
	this.labelsData = $.parseJSON(labelsData) || [];   // Node labels array
	this.linesSpacing = 1.5;	// Scale of line spacing for the vector graph. The higher this is the farther appart channel lanes wil lbe
	this.barCircleRadius = 2;	// the radius of vector plot circles
}


PCAViewer.prototype.plot = function(fromChan, toChan) {
	/*
	 * Do actual plot. Recieves as input channel interval which should be plotted.
	 * These are optional parameters, and if not present will either fall back to 
	 * last plot configuration or to defaults.
	 */
	// Clear the parent div and recreate svg
	$(this.parentDivID).html('');
	var w = this.width;
	var h = this.height;
	this.root = d3.select(this.parentDivID).append("svg").attr("width", w).attr("height", h);
	if (fromChan == null || fromChan == undefined) {
		this.fromChan = this.fromChan || 0;
	} else {
		this.fromChan = fromChan;
	}
	this.toChan = toChan  || 10;
	// Create different 'g' elements for pie, bars charts and for display text on mouse over
    this.root.selectAll("g").data([
            {id: "pie", x: w / 8, h: h / 2.2},
            {id: "bars", x: w / 2, h: h / 2},
            {id: "text", x: 0.1 * w, h: 0.95 * h}
        ])
        .enter().append("g").attr("id", function (d) {
            return d.id;
        })
        .attr("transform", function (d) {
            return "translate(" + d.x + ", " + d.h + ")"
        })
        
    this.root.select("g#text").append("text").classed("pca-text", true).text("");
	this.drawPieChart(this.root.select("g#pie"))	// Left side
	this.drawVectorsPlot(this.root.select("g#bars"));	// Right side
	this.initComponentSelector();	// Selector component
}


PCAViewer.prototype.drawPieChart = function(root) {
	/*
	 * Draw the left side pie chart visualization. 
	 * @param root: a 'g' element to which this will be added
	 */
	var self = this;
	this.loadFractions(this.fromChan, this.toChan);	// Load fractions data for the given interval
	// Create slices data, and then the angles array in order to plot the pie chart
    var slices = this.fractions.data;
    slices.push(this.hiddenFractionsSum);
    var slice_specs = [];
    var slice_acc = 0;
    for (var i = 0; i < slices.length; i++) {
        slice_specs[i] = {startAngle: slice_acc};
        slice_acc += slices[i];
        slice_specs[i].endAngle = slice_acc - 0.0001;	// Small margin so at least something is draw for very small values
    }
    // Create a path element, with arcs of different angles
    this.piePlot = root.selectAll("path").data(slice_specs).enter();
	this.piePlot.append("path")
			        .style("fill", function (d, i) {
			            return self.colorMap(i);
			        })
			        .attr("d", d3.svg.arc().innerRadius(this.pieInnerRadius).outerRadius(this.pieOuterRadius))	// Actual arc generation
			        .on("mouseover", function (d, i) {
			        	// On mouse over update text displayed with new values
			        	// Also update the style to engross the stroke of component over which we hover.
			            var u = self.fractions.data
			            var txt;
			            if (i < (self.toChan - self.fromChan)) {
			            	// We are part of the selected components. So just read the data
			                var ord = tv.util.ord_nums[self.fromChan + i + 1]
			                var v = u[i] * 50 / Math.PI;	// Transform back to % from radians
			                txt = ord.slice(0, 1).toUpperCase() + ord.slice(1, ord.length)
			                    + " component explains " + v.toPrecision(3) + " % of the variance.";
			            }
			            else {
			            	// We just display the fraction for components left out of select
			                txt = "Other " + (self.nComp - (self.toChan - self.fromChan)) 
			                	+ " components explain " + (self.hiddenFractionsSum * 50 / Math.PI).toPrecision(3) + " % of the variance.";
			            }
			            self.root.select("g#text").select("text").text(txt);
			            self.axesPlot.select("g").style("stroke-width", function (e, j) {
			            	// Stroke the coresponding line from the vector plot
			                return i === j ? 3 : 1;
			            });
			        })
			        .on("mouseout", function (d, i) {
			        	// Reset things from mouseover
			            self.root.select("g#text").select("text").text("");
			            self.axesPlot.select("g").style("stroke-width", 1);
			        });
}


PCAViewer.prototype.drawVectorsPlot = function(root) {
	/*
	 * Draw the right side visualization, having weights for individual nodes.
	 */
	var self = this;
	this.loadWeights(this.fromChan, this.toChan);	// Make sure weights loaded and up to date
	var nrNodes = this.nComp;
	var w = this.width;
    var h = this.height;
    var linesSpace = this.linesSpacing;
    
    // Create the labels for axis
    var labels = root.append("g").selectAll("text").data(tv.ndar.range(nrNodes).data).enter();
    labels.append("text")
    				.classed("node-labels", true)
	                .attr("transform", function (d) {
	                    return "translate(" + (w / linesSpace / nrNodes) * (d - nrNodes / 3) + ", " + -h / 2.6 + ") rotate(-60) ";
	                })
	                .text(function (d) {
	                	if (self.labelsData.length > 0) {
	                		return self.labelsData[parseInt(d)];
	                	}
	                    return "node " + d;
	                })

	var lineVertTranslate = (-h / 3); // Translate vertical to make room for labels
	var gridHeight = 2 * h / 3; // The height of the vector grid
    // draw vertical grid lines
    var gridLines = root.append("g").attr("transform", "translate(" + -w / 4.5 + "," + lineVertTranslate + ")")
    									.selectAll("line").data(tv.ndar.range(nrNodes).data).enter();
    gridLines.append("line").attr("x1",function (d) { return w / linesSpace / nrNodes * d })
    						.attr("x2", function (d) { return w / linesSpace / nrNodes * d })
        					.attr("y2", gridHeight)
        					.attr("y1", 0)
        					.attr("stroke", function (d) { return d % 5 === 0 ? "#ccc" : "#eee"; });

    // process the raw weights data to display as separate channels
    var weightsRaw = this.weightsRaw;
    var weightsData = [];
    var weightsScl = Math.max(-weightsRaw.min(), weightsRaw.max());
    var nrOfLines = (this.toChan - this.fromChan);

    for (var i = 0; i < (this.toChan - this.fromChan); i++) {
        weightsData[i] = [];
        for (var j = 0; j < this.nComp; j++) {
        	var normalizedValue = weightsRaw.data[i * this.nComp + j] / weightsScl;
            weightsData[i][j] = [ j * w / linesSpace / nrNodes, 
            					  (gridHeight / nrOfLines / 2) * normalizedValue
            					  ];
        }
    }
	
	var channelHeight = gridHeight / nrOfLines; // This is how much should separate each channel
	var plotTranslation = -gridHeight / 2 + channelHeight / 2; // Translation to get from center to top of plot
    // setup axes groups
    this.axesPlot = root.append("g").selectAll("g").data(weightsData).enter()
    						.append("g").attr("transform", function (d, i) {
            return "translate(" + -w / 4.5 + ", " + (plotTranslation + i * channelHeight) + ")";
        });
    // add zero lines
    this.axesPlot.append("line").attr("x2", w / linesSpace).style("stroke", "black");
    // now go through all groups and add circles for each weights data
    var axes = this.axesPlot[0];
    for (var idx = 0; idx < axes.length; idx++) {
    	d3.select(axes[idx]).append("g")
		    					.attr("fill", "transparent")
							  	.attr("stroke", self.colorMap(idx))
								.selectAll("circle").data(weightsData[idx]).enter().append("circle")
																	.attr("cx", function (d, i) { return d[0]; })
																	.attr("cy", function (d, i) { return -d[1]; }) // Negative x since neg translation bring circle top
																	.attr("r", this.barCircleRadius)
																	.attr("fill", self.colorMap(idx))
    }
    // +- signs
    this.axesPlot.append("text").classed("vt-pm-sign", true).attr("x", -6).text("+")
    this.axesPlot.append("text").classed("vt-pm-sign", true).attr("x", -5).attr("y", 5).text("-")
}


PCAViewer.prototype.loadFractions = function(fromIdx, toIdx) {
	/*
	 * Load the fractions from the given interval, which is represented by [fromIdx, toIdx]
	 */
	var self = this;
	var fractionsDataUrl = this.fractionsDataUrl + '?from_comp=' + fromIdx + ';to_comp=' + toIdx;
	doAjaxCall({
	        async:false,
	        type:'GET',
	        url:fractionsDataUrl,
	        success:function (data) {
	        	data = $.parseJSON(data);
	        	self.fractions = tv.ndar.from(data.slice(0, data.length - 1)).mul(2 * Math.PI);	// Convert to radians
	        	self.hiddenFractionsSum = data[data.length - 1] * 2 * Math.PI;	// Last element is the sum of the rest as for the datatype method
	        },
	        error: function(x, e) {}
	    });
}


PCAViewer.prototype.loadWeights = function(fromIdx, toIdx) {
	/*
	 * Load the weights for the given interval, which is represented by [fromIdx, toIdx]
	 */
	var self = this;
	var weightsDataUrl = this.weightsDataUrl + '?from_comp=' + fromIdx + ';to_comp=' + toIdx;
	var nrChans = toIdx - fromIdx;
	doAjaxCall({
	        async:false,
	        type:'GET',
	        url:weightsDataUrl,
	        success:function (data) {
	        	self.weightsRaw = tv.ndar.from($.parseJSON(data));
	        	self.nComp = self.weightsRaw.length() / nrChans;
	        },
	        error: function(x, e) {}
	    });
}


PCAViewer.prototype.initComponentSelector = function() {
	/*
	 * Initialize the selector for the currently displayed components
	 */
	var self = this;
	var firstChanSelect = $('#first-chan-selector');
	var secondChanSelect = $('#second-chan-selector');
	// Reset old options
	firstChanSelect.html('');
	secondChanSelect.html('');
	// Add option for each component
	for (var i = 0; i < this.nComp; i++) {
		var oFrom = new Option(i + 1, i);
		$(oFrom).html(i);
		firstChanSelect.append(oFrom);
		var oTo = new Option(i + 1, i);
		$(oTo).html(i);
		secondChanSelect.append(oTo);
	}
	// Mark as selected the currently set interval
	firstChanSelect.val(this.fromChan);
	secondChanSelect.val(this.toChan);
	// On click, just update the plot
	document.getElementById("refresh-channels").onclick = function () {
																		var fromIdx = parseInt(firstChanSelect.val());
																		var toIdx = parseInt(secondChanSelect.val());
																		if (fromIdx == toIdx) { toIdx = toIdx + 1 }
																		if (fromIdx < toIdx) {
																			self.plot(fromIdx, toIdx);
																		} else {
																			self.plot(toIdx, fromIdx);
																		}
																	}
}
