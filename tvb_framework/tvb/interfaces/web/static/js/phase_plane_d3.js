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
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

/* globals d3 */

var TVBUI = TVBUI || {};

/**
 * @module d3 phase plane
 */
(function(){
    "use strict";
    // from colorbrewer dark2
    var trajColors = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"];

    // The logical coordinate space. All dimensions are in this space. Mapping to screen space is done by svg keeping aspect ration.
    var viewBox = "0 0 1000 1000";
    var planeWidth = 800;
    var planeHeight = 800;

    function GraphBase(selector){
        var self = this;
        // --- declarations and global structure ---
        // We create the svg dom in js. Alternatively this could be written declarative in svg and retrieved here by .select()
        this.svg = d3.select(selector).attr('viewBox', viewBox);

        this.svg.append("clipPath").attr("id", "clip")      // clip all phase plane geometry
            .append("rect")
              .attr("width", planeWidth)
              .attr("height", planeHeight);

        this.svg.append("text")     // title
            .attr("x", 500)
            .attr("y", 20)
            .attr("class", "title")
            .text("Phase plane");

        // --- phase plane structure ---
        this.plane_with_axis = this.svg.append('g')     // groups phase plane, axis, labels trajectories and overlay
            .attr("transform", "translate(100, 40)");

        this.plane_g = this.plane_with_axis.append('g') // the vectors are drawn here
            .attr('class', 'phasePlane')
            .attr("clip-path", "url(#clip)");

        this.xAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis')
            .attr("transform", "translate(0, 800)");

        this.yAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis');

        this.xLabel = this.plane_with_axis.append("text")
            .attr("class", "axislabel")
            .attr("x", planeWidth + 20)
            .attr("y", planeHeight);

        this.yLabel = this.plane_with_axis.append("text")
            .attr("class", "axislabel")
            .attr("text-anchor", "end")
            .attr("y", -15);

        this.vectorAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis')
            .attr("transform", "translate(825, 300)");

        this.lineBuilder = d3.svg.line()
                .x(function(d) { return self.xScale(d[0]); })
                .y(function(d) { return self.yScale(d[1]); })
                .interpolate("linear");

        this.trajectories_g = this.plane_with_axis.append('g')
            .attr('class', 'traj')
            .attr("clip-path", "url(#clip)");
    }

    GraphBase.prototype.setLabels = function(xlabel, ylabel) {
        this.xLabel.text(xlabel);
        this.yLabel.text(ylabel);
    };

    GraphBase.prototype._computePhasePlaneScales = function(data){
        var xrange = d3.extent(data, function (d) {return d[0];});
        var yrange = d3.extent(data, function (d) {return d[1];});
        if (yrange[0] === yrange[1]){
            // signal is constant. The scale would become singular. So create a fake range.
            var delta = yrange[0]/4;
            yrange[0] -= delta;
            yrange[1] += delta;
        }
        this.xScale = d3.scale.linear().domain(xrange).range([0, planeWidth]);
        this.yScale = d3.scale.linear().domain(yrange).range([planeHeight, 0]);  // reverse range to compensate 4 y axis direction
    };

    /**
     * @constructor
     * @extends RegionSelectComponent
     */
    function PhasePlane(selector){
        GraphBase.call(this, selector);
        var self = this;
        this.VECTOR_RANGE = 0.80;
        this.onClick = function(){};

        this.svg.append("clipPath").attr("id", "clip_plot")
            .append("rect")
              .attr("width", planeWidth)
              .attr("height", 200);

        this.plane_with_axis.append('text')
            .attr('class', 'axis')
            .text('arrow scale')
            .attr("transform", "translate(880, 260) rotate(90)");

        var overlay = this.plane_with_axis.append("rect")   // this is a transparent rect used for receiving mouse events
            .attr("class", "overlay")
            .attr("pointer-events", "all")
            .attr("width", planeWidth)
            .attr("height", planeHeight)
            .on("click", function(){
                var xy = d3.mouse(this);    // ask for mouse position relative to parent
                var x = self.xScale.invert(xy[0]);  // translate position to data space
                var y = self.yScale.invert(xy[1]);
                self.onClick(x, y);
            });

        // --- signals plot ---
        this.plot_with_axis = this.svg.append('g')
            .attr("transform", "translate(100, 880)");

        this.plot_nodata = this.svg.append('text')
            .attr("class", "title")
            .attr("x", 500)
            .attr("y", 920)
            .text('click in the phase plane to show signals for the last trajectory')
            .attr('display', 'none');

        this.plot_g = this.plot_with_axis.append('g')
            .attr('class', 'traj')
            .attr("clip-path", "url(#clip_plot)");

        this.plot_with_axis.append("text")
            .attr("class", "axislabel")
            .text("time[ms]")
            .attr("x", planeWidth + 10)
            .attr("y", 120);

        this.xAxis_plot_g = this.plot_with_axis.append('g').attr('class', 'axis').attr("transform", "translate(0, 100)");
        this.yAxis_plot_g = this.plot_with_axis.append('g').attr('class', 'axis');
        this.plot_legend_g = this.plot_with_axis.append('g').attr("transform", "translate(820, 0)");
    }

    // proto chain setup PhasePlane.prototype = {new empty obj} -> GraphBase.prototype
    // Object.create is needed PhasePlane.prototype = GraphBase.prototype;
    // would have had the effect that changing PhasePlane.prototype would've changed GraphBase.prototype
    PhasePlane.prototype = Object.create(GraphBase.prototype);

    /**
     * Computes
     * this.xScale, this.yScale : mapping vector origins to screen positions
     * this.vectorScale : mapping vector magnitude to reasonable line lengths
     */
    PhasePlane.prototype._computePhasePlaneScales = function(data){
        // call base version
        GraphBase.prototype._computePhasePlaneScales.call(this, data);
        // scale to normalize the vector lengths
        // Maps from symmetric to symmetric range to ensure that 0 maps to 0.
        // This scaling avoids a mess of lines for big vectors but the lengths of the vectors
        // are not absolute nor compatible with the x, y axis.
        var dxmax = d3.max(data, function(d){return Math.abs(d[2]);});
        var dymax = d3.max(data, function(d){return Math.abs(d[3]);});
        var max_delta = Math.max(dxmax, dymax);
        this.vectorScale = d3.scale.linear().domain([-max_delta, max_delta]).range([-this.VECTOR_RANGE/2, this.VECTOR_RANGE/2]);
    };

    /**
     * Draws the phase plane, vector quiver nullclines and axis
     * @param data a object {plane: [[x, y, dx, dy], ...], nullclines: [[x,y], ...]}
     */
    PhasePlane.prototype.draw = function(data){
        var self = this;
        this._computePhasePlaneScales(data.plane);

        // axes
        var xAxis = d3.svg.axis().scale(this.xScale).orient('bottom');
        var yAxis = d3.svg.axis().scale(this.yScale).orient("left");
        var vectorAxis = d3.svg.axis().scale(this.vectorScale).orient("right").ticks(5);
        this.xAxis_g.transition().call(xAxis);
        this.yAxis_g.transition().call(yAxis);
        this.vectorAxis_g.transition().call(vectorAxis);

        // vectors
        var p = this.plane_g.selectAll('line').data(data.plane);
        p.enter().append('line');

        // update
        p.transition()
            .attr('x1', function(d){
                return self.xScale(d[0]);
            })
            .attr('y1', function(d){
                return self.yScale(d[1]);
            })
            .attr('x2', function(d){
                // note that this scaling no longer ensures that the maximal vector has a reasonable size in pixels
                // Previous implementation did that but buggy: vector orientations were bad
                // todo: properly scale vectors so that the screen length is reasonable
                // self.vectorScale maps to a fixed hard coded interval
                var scaled_vector = self.vectorScale(d[2]);
                return self.xScale(d[0] + scaled_vector);
            })
            .attr('y2', function(d){
                var scaled_vector = self.vectorScale(d[3]);
                return self.yScale(d[1] + scaled_vector);
            });

        // nullclines
        var nc = this.plane_g.selectAll('path').data(data.nullclines);
        nc.enter().append('path');
        nc.attr('d', function(d){
                return self.lineBuilder(d.path);
            })
            .attr('stroke', function(d){
                return ['#d73027', '#1a9850'][d.nullcline_index];
            })
            .attr('class', 'traj');
        nc.exit().remove();
    };

    /**
     * Draws a family of trajectories
     * @param trajectories. array shaped trajectory, point, xy [[[x,y], ...], ...] in the same space as the vectors!
     */
    PhasePlane.prototype.drawTrajectories = function(trajectories){
        var self = this;
        var p = this.trajectories_g.selectAll('path').data(trajectories);
        p.enter().append('path');
        p.exit().remove();
        p.attr('d', function(d){
                return self.lineBuilder(d);
            })
            .attr('stroke', function(d, i){
                return trajColors[i  % trajColors.length];
            });
    };

    PhasePlane.prototype._computePlotScales = function(signal){
        // compute scale
        var xrange = d3.extent(signal[0], function (d) {return d[0];});
        var yrange = d3.extent(signal[0], function (d) {return d[1];});

        for (var i = 1; i < signal.length; i++) {
            var current = d3.extent(signal[i], function (d) {return d[1];});
            yrange[0] = Math.min(yrange[0], current[0]);
            yrange[1] = Math.max(yrange[1], current[1]);
        }

        var xS = d3.scale.linear().domain(xrange).range([0, planeWidth]);
        var yS = d3.scale.linear().domain(yrange).range([100, 0]);
        return [xS, yS];
    };

    /**
     * Draws the state variable signals. It is used to draw the signals for the last trajectory.
     * @param signal [[x,y], ... ] for each state variable
     */
    PhasePlane.prototype.drawSignal = function(signal){
        var self = this;

        if (signal.length !== 0) {
            var scales = this._computePlotScales(signal);
            var xAxis = d3.svg.axis().scale(scales[0]).orient('bottom');
            var yAxis = d3.svg.axis().scale(scales[1]).orient('left').ticks(5);

            this.xAxis_plot_g.call(xAxis);
            this.yAxis_plot_g.call(yAxis);
            this.plot_nodata.attr('display', 'none');
            this.plot_with_axis.attr('display', null);

            var lineBuilder = d3.svg.line()
                .x(function (d) {
                    return scales[0](d[0]);
                })
                .y(function (d) {
                    return scales[1](d[1]);
                })
                .interpolate("linear");
            var colorS = d3.scale.category10().domain(d3.range(signal.length));
        }else{
            this.plot_nodata.attr('display', null);
            this.plot_with_axis.attr('display', 'none');
        }

        var p = this.plot_g.selectAll('path').data(signal);
        p.enter().append('path');
        p.exit().remove();

        p.attr('d', function(d){
                return lineBuilder(d);
            })
            .attr('stroke', function(d, i){
                return colorS(i);
            });
    };


    PhasePlane.prototype.setPlotLabels = function (labels){
        var colorS = d3.scale.category10().domain(d3.range(labels.length));
        var labels_el = this.plot_legend_g.selectAll('text').data(labels);
        labels_el.enter().append('text');
        labels_el.exit().remove();
        labels_el.attr('transform', function(d, i){
                return 'translate(0, ' + i * 20 + ')';
            })
            .attr('fill', function(d, i){
                return colorS(i);
            })
            .text(function(d){return d;});
    };


    function PhaseGraph(selector){
        GraphBase.call(this, selector);
        var self = this;
        this.VECTOR_RANGE = 40;
        this.xZeroAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axisZero');
        this.phaseLine_g = this.plane_with_axis.append('g')
            .attr('class', 'phaseLine');
        this.plane_with_axis.append('line').attr('x1','0').attr('x2','0').attr('y1','0').attr('y2','800');
        this.trajectories_g.attr('stroke', trajColors[3]);
        this.vectorAxis_g.attr('transform', 'translate(850, 10)');
    }
    // proto chain setup
    PhaseGraph.prototype = Object.create(GraphBase.prototype);

    PhaseGraph.prototype._computePhasePlaneScales = function(data){
        GraphBase.prototype._computePhasePlaneScales.call(this, data);
        this.phaseScale = this.xScale.copy();
        this.phaseScale.range([planeHeight, 0]);
        var max_delta  = d3.max(data, function(d){return Math.abs(d[1]);});
        this.phaseVectorScale = d3.scale.linear().domain([-max_delta, max_delta]).range([-this.VECTOR_RANGE/2, this.VECTOR_RANGE/2]);
    };

    PhaseGraph.prototype.draw = function(data){
        var self = this;
        this._computePhasePlaneScales(data.signal);
        // draw x' vs x graph
        var xAxis = d3.svg.axis().scale(this.xScale).orient('bottom');
        var yAxis = d3.svg.axis().scale(this.yScale).orient("left");
        this.xZeroAxis_g.transition().call(xAxis).attr("transform", "translate(0, "+ this.yScale(0) + ")");
        this.xAxis_g.transition().call(xAxis);
        this.yAxis_g.transition().call(yAxis);

        var p = this.trajectories_g.selectAll('path').data([data.signal]);
        p.enter().append('path');
        p.transition().attr('d', self.lineBuilder(data.signal));
        p.exit().remove();

        // vectors
        // sub sample
        var vector_data = [];
        for (var i = 0; i < data.signal.length; i+=4){
            vector_data.push(data.signal[i]);
        }
        var ph = this.phaseLine_g.selectAll('line').data(vector_data);
        ph.enter().append('line')
            .attr('x1', 850).attr('x2', 850);

        // update
        ph.transition()
            .attr('y1', function(d){
                return self.phaseScale(d[0]);
            })
            .attr('y2', function(d){
                return self.phaseScale(d[0]) - self.phaseVectorScale(d[1]);
            });

        var phaseLineAxis = d3.svg.axis().scale(this.phaseScale).orient("right").ticks(5);
        this.vectorAxis_g.transition().call(phaseLineAxis);

        var z = this.phaseLine_g.selectAll('circle').data(data.zeroes);
        z.enter().append('circle');
        z.attr('r', 8)
            .attr('cy', function(d){return self.phaseScale(d);})
            .attr('cx', 850)
            .attr('fill', trajColors[5]);
        z.exit().remove();
    };


    TVBUI.PhasePlane = PhasePlane;
    TVBUI.PhaseGraph = PhaseGraph;
})();
