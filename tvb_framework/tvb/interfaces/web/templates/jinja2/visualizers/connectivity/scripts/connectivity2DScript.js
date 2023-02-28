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

// with this factor will be changed the weights of the edges of all charts
var scaleFactor = 5;
// the index of the node which should be the root of the chart which contains the both hemispheres
var rootNode = 1;
// contains the points that were selected by the user. Only this points will be drawn in the main view.
// If the list is empty the all the points will be drawn in the main view.
var selectedPoints = [];
// used to see if the user switched to the view with disks of different colors and size
var shouldDisplayCustomView = false;

var C2D_selectedView = 'both';
var C2D_shouldRefreshNodes = true;
var C2D_canvasDiv = 'hemispheresDisplay';

var rgraph;         // keep the rgraph global so that it can be accessed when exporting images from canvas
var exportScale;    // used for resizing the plot on exporting from canvas

var C2D_hemispheresJSON;

/**
 * Used for drawing a chart from the given json
 *
 * @param json a string which contains all the data needed for drawing a chart
 * @param shouldRefreshNodes <code>true</code> if the drawn chart contains data related to both hemisphere
 */
function drawConnectivity(json, shouldRefreshNodes) {
    if (!rgraph) {      // make the initialisations
        rgraph = new $jit.RGraph({
            'injectInto': C2D_canvasDiv,
            Node: {
                'overridable': true,
                'alpha': 0.7
            },
            Edge: {
                'overridable': true,
                'color': '#cccc00'
            },
            //Set polar interpolation. Default's linear.
            interpolation: 'polar',
            //This method is called right before plotting an edge. This method is useful to change edge styles individually.
            onBeforePlotLine: function(adj) {
                if (!adj.data.$lineWidth)
                    adj.data.$lineWidth = Math.max(Math.min(adj.data.weight * scaleFactor, 12.0), 0); // clamp to 0..12
            },
            //Add node click handler and some styles. This method is called only once for each node/label crated.
            onCreateLabel: function(domElement, node) {
                domElement.innerHTML = node.name;
                var style = domElement.style;
                style.cursor = 'default';
                style.fontSize = "0.8em";
                style.color = "#fff";
            },
            //This method is called when rendering/moving a label.
            //This is method is useful to make some last minute changes to node labels like adding some position offset.
            onPlaceLabel: function(domElement) {
                var style = domElement.style;
                var left = parseInt(style.left);
                var w = domElement.offsetWidth;
                style.left = (left - w / 2) + 'px';
            },
            onBeforePlotNode: function(node) {
                if (shouldDisplayCustomView) {
                    node.data.$dim = node.data.customShapeDimension;
                    node.data.$color = node.data.customShapeColor;
                }
            },

            onAfterCompute: function() {
            }

        });
        // interface-like methods needed for exporting HiRes images
        rgraph.canvas.canvases[0].canvas.drawForImageExport = __resizeCanvasBeforeExport;
        rgraph.canvas.canvases[0].canvas.afterImageExport   = __restoreCanvasAfterExport;

    }
    if(!json){
        rgraph.loadJSON([{id:'root', name:'THIS HEMISPHERE IS EMPTY',
                          data: {radius: 0, angle: 0} }]);
    } else {
        var first_json_node = hasNodesInJson(json);
        if (first_json_node >= 0) {
            rootNode = first_json_node;

            //load graph.
            if (shouldRefreshNodes) {
                rgraph.loadJSON(json, rootNode);
            } else {
                rgraph.loadJSON(json, 1);
            }
            // remove all the nodes that were not selected by the user
            rgraph.graph.eachNode(function (node) {
                if (shouldRefreshNodes && selectedPoints.length > 0 && !isNodeSelected(node.id)) {
                    rgraph.graph.removeNode(node.id);
                }
            });
        } else {
            rgraph.loadJSON([{
                id: 'root', name: 'SELECT AT LEAST ONE NODE FROM THIS REGION TO REVEAL CONNECTIONS...',
                data: {radius: 0, angle: 0}
            }]);
        }
    }

    //compute positions and plot
    rgraph.refresh();
    rgraph.plot();
    rgraph.controller.onBeforeCompute(rgraph.graph.getNode(rgraph.root));
    rgraph.controller.onAfterCompute();
}

/**
 * Show/hide aasociated info in 2D displays.
 */
function change2DView() {
    shouldDisplayCustomView = !shouldDisplayCustomView;
	var btn = document.getElementById("change2DViewBtn");
    if (shouldDisplayCustomView) {
        btn.innerHTML ="Hide Details";
        btn.className = "action action-minus";
    } else {
        btn.innerHTML = "Show All";
        btn.className = "action action-plus";
    }
	
	C2D_displaySelectedPoints();
}

/**
 * Each time when the user check/uncheck a check box this method will be called to redraw the main view.
 */
function C2D_displaySelectedPoints() {
    selectedPoints = [];
    rootNode = parseInt(GVAR_interestAreaNodeIndexes[0]);
	for (var i = 0; i < GVAR_interestAreaNodeIndexes.length; i++) {
		selectedPoints.push(GVAR_pointsLabels[GVAR_interestAreaNodeIndexes[i]]);
	}
	
    drawConnectivity(C2D_hemispheresJSON[C2D_selectedView], C2D_shouldRefreshNodes);
}

/**
 * Resize the jit-created canvas using its library functions
 * @private
 */
function __resizeCanvasBeforeExport() {
    var size = rgraph.canvas.getSize();
    exportScale = C2I_EXPORT_HEIGHT / size.height;
    rgraph.canvas.resize(size.width * exportScale, size.height * exportScale);
    rgraph.canvas.scale(exportScale, exportScale);
}

/**
 * After canvas export, bring the canvas back to original size
 * @private
 */
function __restoreCanvasAfterExport() {
    var size = rgraph.canvas.getSize();
    exportScale = 1 / exportScale;
    rgraph.canvas.resize(size.width * exportScale, size.height * exportScale);
}

/**
 * Check if any of the selected nodes are present in the json.
 */
function hasNodesInJson(json) {
	for (var i=0; i < json.length; i++) {
		if (isNodeSelected(json[i].id)) {
			return i;
		}
	}
	return -1;
}


/**
 * Checks to see if a certain node was selected by the user
 *
 * @param nodeId the id of the node to check. (A label of a point from the connectivity matrix)
 */
function isNodeSelected(nodeId) {
    var elemIdx = $.inArray(nodeId, selectedPoints);
    return elemIdx != -1;
}

function prepareConnectivity2D(left_json, full_json, right_json) {
    C2D_hemispheresJSON = {
        left:left_json,
        both:full_json,
        right:right_json
    };
}
