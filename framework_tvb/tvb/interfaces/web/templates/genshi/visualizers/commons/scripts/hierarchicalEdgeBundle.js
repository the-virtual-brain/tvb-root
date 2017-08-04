/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
 * A function for drawing a hierarchical edge bundle
 *
 * @param data    The data structure that has the region labels and the attributes matrix
 * @param svg_d3  The html svg element that is going to be mutated
 */
function init_data(data, test) {

    var l = data.region_labels.length;

    var svg_d3 = data.svg.d3;

    var jsonified_region_labels = [];

    for (var i = 0; i < l; i++) {
        var json_line = {};
        json_line.imports = [];
        var k = 0; //k is a counter for connected regions with the j-th region
        for (var j = 0; j < l; j++) {
            var w = 0;
            w = data.matrix[i * l + j];
            json_line.name = data.region_labels[i];
            if (test(w)) {
                json_line.imports[k] = data.region_labels[j];
                k++;
            }
        }
        jsonified_region_labels[i] = json_line;
    }

    var diameter = data.svg.svg.height(),
        radius = diameter * 0.51 + 1,
        innerRadius = radius - 150;

    var cluster = d3.cluster()
        .size([360, innerRadius]);

    var line = d3.radialLine()
        .curve(d3.curveBundle.beta(0.85))
        .radius(function (d) {
            return d.y;
        })
        .angle(function (d) {
            return d.x / 180 * Math.PI;
        });

    var svg = svg_d3
        .append("g")
        .attr("transform", "translate(" + radius + "," + radius + ")");

    var link = svg.append("g").selectAll(".link"),
        node = svg.append("g").selectAll(".node");

    var root = packageHierarchy(jsonified_region_labels)
        .sum(function (d) {
            return d.size;
        });

    cluster(root);

    link = link
        .data(packageImports(root.leaves()))
        .enter().append("path")
        .each(function (d) {
            d.source = d[0], d.target = d[d.length - 1];
        })
        .attr("class", "link")
        .attr("d", line);

    node = node
        .data(root.leaves())
        .enter().append("text")
        .attr("class", "node")
        .attr("dy", "0.31em")
        .attr("transform", function (d) {
            return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 8) + ",0)" + (d.x < 180 ? "" : "rotate(180)");
        })
        .attr("text-anchor", function (d) {
            return d.x < 180 ? "start" : "end";
        })
        .text(function (d) {
            return d.data.key;
        })
        .on("mouseover", mouseovered)
        .on("mouseout", mouseouted);

    function mouseovered(d) {
        node
            .each(function (n) {
                n.target = n.source = false;
            });

        link
            .classed("link-target", function (l) {
                if (l.target === d) return l.source.source = true;
            })
            .classed("link-source", function (l) {
                if (l.source === d) return l.target.target = true;
            })
            .filter(function (l) {
                return l.target === d || l.source === d;
            })
            .raise();

        node
            .classed("node-target", function (n) {
                return n.target;
            })
            .classed("node-source", function (n) {
                return n.source;
            });
    }

    function mouseouted(d) {
        link
            .classed("link-target", false)
            .classed("link-source", false);

        node
            .classed("node-target", false)
            .classed("node-source", false);
    }

// Lazily construct the package hierarchy from class names.
    function packageHierarchy(classes) {
        var map = {};

        function find(name, data) {
            var node = map[name], i;
            if (!node) {
                node = map[name] = data || {name: name, children: []};
                if (name.length) {
                    node.parent = find(name.substring(0, i = name.lastIndexOf(".")));
                    node.parent.children.push(node);
                    node.key = name.substring(i + 1);
                }
            }
            return node;
        }

        classes.forEach(function (d) {
            find(d.name, d);
        });

        return d3.hierarchy(map[""]);
    }

// Return a list of imports for the given array of nodes.
    function packageImports(nodes) {
        var map = {},
            imports = [];

        // Compute a map from name to node.
        nodes.forEach(function (d) {
            map[d.data.name] = d;
        });

        // For each import, construct a link from the source to target node.
        nodes.forEach(function (d) {
            if (d.data.imports) d.data.imports.forEach(function (i) {
                imports.push(map[d.data.name].path(map[i]));
            });
        });

        return imports;
    }
}
