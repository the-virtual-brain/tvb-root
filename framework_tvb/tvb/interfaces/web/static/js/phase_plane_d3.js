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
 * http://www.gnu.org/licenses/old-licenses/gpl-2.
 *
 *   CITATION:
 * When using The Virtual Brain for scientific publications, please cite it as follows:
 *
 *   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
 *   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
 *       The Virtual Brain: a simulator of primate brain network dynamics.
 *   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
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
    // from colorbrewer
    var dark2 = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"];
    var spectral = ["#9e0142","#d53e4f","#f46d43","#fdae61","#fee08b","#ffffbf","#e6f598","#abdda4","#66c2a5","#3288bd","#5e4fa2"];

    // The logical coordinate space. All dimensions are in this space. Mapping to screen space is done by svg keeping aspect ration.
    var viewBox = "0 0 1000 1000";
    var planeWidth = 800;
    var planeHeight = 800;

    function PhasePlane(onClick){
        var self = this;
        this.onClick = onClick;
        this.trajs = [];        // keeps the trajectories/signals raw data
        this.signals = [];
        // --- declarations and global structure ---
        this.svg = d3.select('.dynamicChart').attr('viewBox', viewBox);

        this.svg.append("clipPath").attr("id", "clip")      // clip all phase plane geometry
            .append("rect")
              .attr("width", planeWidth)
              .attr("height", planeHeight);

        this.svg.append("clipPath").attr("id", "clip_plot")
            .append("rect")
              .attr("width", planeWidth)
              .attr("height", 200);

        this.svg.append("text")     // title
            .attr("x", 500)
            .attr("y", 10)
            .attr("font-size","30")
            .attr("text-anchor", "middle")
            .text("Phase plane");

        // --- phase plane structure ---
        this.plane_with_axis = this.svg.append('g')     // groups phase plane, axis, labels trajectories and overlay
            .attr("transform", "translate(100, 20)");

        this.plane_g = this.plane_with_axis.append('g') // the vectors are drawn here
            .attr('class', 'phasePlane')
            .attr("clip-path", "url(#clip)");

        this.xAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis')
            .attr("transform", "translate(0, 800)");
        this.yAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis');
        this.vectorAxis_g = this.plane_with_axis.append('g')
            .attr('class', 'axis')
            .attr("transform", "translate(825, 300)");

        this.xLabel = this.plane_with_axis.append("text")
            .attr("class", "xlabel")
            .attr("x", planeWidth + 20)
            .attr("y", planeHeight);

        this.yLabel = this.plane_with_axis.append("text")
            .attr("class", "ylabel")
            .attr("text-anchor", "end")
            .attr("y", -15);

        this.trajectories_g = this.plane_with_axis.append('g')
            .attr('class', 'traj')
            .attr("clip-path", "url(#clip)");

        var overlay = this.plane_with_axis.append("rect")   // this is a transparent rect used for receiving mouse events
            .attr("class", "overlay")
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
            .attr("transform", "translate(100, 860)");

        this.plot_g = this.plot_with_axis.append('g')
            .attr('class', 'traj')
            .attr("clip-path", "url(#clip_plot)");

        this.xAxis_plot_g = this.plot_with_axis.append('g').attr('class', 'axis').attr("transform", "translate(0, 100)");
        this.yAxis_plot_g = this.plot_with_axis.append('g').attr('class', 'axis');

        this.lineBuilder = d3.svg.line()
                .x(function(d) { return self.xScale(d[0]); })
                .y(function(d) { return self.yScale(d[1]); })
                .interpolate("linear");
    }

    PhasePlane.prototype.colors = dark2;

    /**
     * Computes
     * this.xScale, this.yScale : mapping vector origins to screen positions
     * this.vectorScale : mapping vector magnitude to reasonable line lengths
     */
    PhasePlane.prototype._computePhasePlaneScales = function(data){
        var xrange = d3.extent(data, function(d){return d[0];});
        var yrange = d3.extent(data, function(d){return d[1];});
        this.xScale = d3.scale.linear().domain(xrange).range([0, planeWidth]);
        this.yScale = d3.scale.linear().domain(yrange).range([planeHeight, 0]);  // reverse range to compensate 4 y axis direction
        // scale to normalize the vector lengths
        // Maps from symmetric to symmetric range to ensure that 0 maps to 0.
        // This scaling avoids a mess of lines for big vectors but the lengths of the vectors
        // are not absolute nor compatible with the x, y axis.
        var dxmax = d3.max(data, function(d){return Math.abs(d[2]);});
        var dymax = d3.max(data, function(d){return Math.abs(d[3]);});
        var max_delta = Math.max(dxmax, dymax);
        this.vectorScale = d3.scale.linear().domain([-max_delta, max_delta]).range([-50, 50]);
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
        this.xAxis_g.call(xAxis);
        this.yAxis_g.call(yAxis);
        this.vectorAxis_g.call(vectorAxis);

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
                return self.xScale(d[0]) + self.vectorScale(d[2]);
            })
            .attr('y2', function(d){
                return self.yScale(d[1]) - self.vectorScale(d[3]); // - for compatibility with the y scale
            });

        // nullclines
        var nc = this.plane_g.selectAll('path').data(data.nullclines);
        nc.enter().append('path');
        nc.transition()
            .attr('d', function(d){
                return self.lineBuilder(d);
            })
            .attr('stroke', function(d, i){
                return ['#d73027', '#1a9850'][i];
            })
            .attr('class', 'traj');
    };

    /**
     * Adds a new trajectory to the list of trajectories to be drawn and draws them all.
     * @param data [[x,y], ...] in the same space as the vectors!
     */
    PhasePlane.prototype.drawTrajectory = function(data){
        var self = this;
        this.trajs.push(data);

        var p = this.trajectories_g.selectAll('path').data(this.trajs);
        p.enter().append('path');
        p.exit().remove();
        p.attr('d', function(d){
                return self.lineBuilder(d);
            })
            .attr('stroke', function(d, i){
                return self.colors[i  % self.colors.length];
            });
    };

    PhasePlane.prototype.drawSignal = function(data){
        var self = this;
        this.signals.push(data);

        var xrange = d3.extent(data, function(d){return d[0];});
        var yrange = d3.extent(data, function(d){return d[1][0];});
        var xS = d3.scale.linear().domain(xrange).range([0, 800]);
        var yS = d3.scale.linear().domain(yrange).range([100, 0]);

        var xAxis = d3.svg.axis().scale(xS).orient('bottom');
        var yAxis = d3.svg.axis().scale(yS).orient("left");
        this.xAxis_plot_g.call(xAxis);
        this.yAxis_plot_g.call(yAxis);

        var lineFn = d3.svg.line()
                        .x(function(d) { return xS(d[0]); })
                        .y(function(d) { return yS(d[1][0]); })
                        .interpolate("linear");

        var p = this.plot_g.selectAll('path').data(this.signals);
        p.enter().append('path');
        p.exit().remove();
        p.attr('d', function(d){
                return lineFn(d);
            })
            .attr('stroke', function(d, i){
                return self.colors[i  % self.colors.length];
            });
    };

    PhasePlane.prototype.setLabels = function(xlabel, ylabel) {
        this.xLabel.text(xlabel);
        this.yLabel.text(ylabel);
    };

    PhasePlane.prototype.clearTrajectories = function(){
        this.trajs = [];
        this.signals = [];
        this.drawTrajectory([]);
        this.drawSignal([]);
    };

    TVBUI.PhasePlane = PhasePlane;
})();
