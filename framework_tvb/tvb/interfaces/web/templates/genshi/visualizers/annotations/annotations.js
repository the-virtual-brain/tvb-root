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

/**
 * Depends on the following GLOBALS: gl, BASE_PICK, ColSch, LEG, HLPR_readJSONfromFile
 *
 * @param baseUrl - Current web installation base URL (is needed for JSTree style URLS)
 * @param treeDataUrl - URL from where the Annotations tree will be loaded
 * @param triangleToRegionUrl - URL for reading a vector of triangle to connectivity region mapping
 * @param minValue - Minimum value for the colors (used for legend)
 * @param maxValue - Maximum value for the colors (used for legend)
 * @param urlRegionBoundaries - URL for reading Region Mapping boundary lines.
 * @constructor
 */
function ANN_Displayer(baseUrl, treeDataUrl, triangleToRegionUrl, minValue, maxValue, urlRegionBoundaries) {

    this.treeElem = $("#treeStructure");
    this.triaglesMappings = [];
    this.regionToTriangleMapping = {};
    this.selectFrom3DMode = false;
    this.prefixNodeIdTVB = "node_tvb_";
    this.prefixNodeIdBRCO = "node_brco_";

    this._init = function (baseUrl, treeDataUrl, triangleToRegionUrl, minValue, maxValue) {

        this._populateAnnotationsTree(baseUrl, treeDataUrl);

        ColSch_initColorSchemeGUI(minValue, maxValue);
        LEG_initMinMax(minValue, maxValue);
        LEG_generateLegendBuffers();
        BASE_PICK_initLegendInfo(maxValue, minValue);

        this.triaglesMappings = HLPR_readJSONfromFile(triangleToRegionUrl);
        this._prepareRegionToTriangleMapping();

        // TODO TVB-1924
        //initRegionBoundaries(urlRegionBoundaries);

    };

    this._populateAnnotationsTree = function (baseUrl, treeDataUrl) {

        this.treeElem.jstree({
            "plugins": ["themes", "json_data", "ui", "crrm"],
            "themes": {
                "theme": "default",
                "dots": true,
                "icons": true,
                "url": baseUrl + "static/jquery/jstree-theme/style.css"
            },
            "json_data": {
                "ajax": {
                    url: treeDataUrl,
                    success: function (d) {
                        return eval(d);
                    }
                }
            }
        });

        var SELF = this;
        this.treeElem.bind("select_node.jstree", function (event, data) {

            if (SELF.selectFrom3DMode) {
                return;
            }

            var selectedNode = data.rslt.obj.attr("id");
            if (selectedNode && selectedNode.indexOf(SELF.prefixNodeIdTVB) == 0) {
                selectedNode = parseInt(selectedNode.replace(SELF.prefixNodeIdTVB, ''));
                displayMessage("Selected Region " + selectedNode, 'infoMessage');

                TRIANGLE_pickedIndex = parseInt(SELF.regionToTriangleMapping[selectedNode]);
	            BASE_PICK_moveBrainNavigator(true);
            }
        })
    };

    this.selectTreeNode = function () {

        if (TRIANGLE_pickedIndex >= 0) {
            if (TRIANGLE_pickedIndex >= this.triaglesMappings.length) {
                displayMessage("Picked triangle " + TRIANGLE_pickedIndex + " outside of our region mapping", 'warningMessage');
                return;
            }
            var pickedRegion = this.triaglesMappings[TRIANGLE_pickedIndex];
            displayMessage("Picked triangle " + TRIANGLE_pickedIndex + " in region " + pickedRegion, 'infoMessage');

            this.selectFrom3DMode = true;
            this.treeElem.jstree("deselect_all");
            var toBeSelectedId = "#" + this.prefixNodeIdTVB + pickedRegion;
            this.treeElem.jstree("select_node", toBeSelectedId);
            this.treeElem.jstree("open_node", toBeSelectedId);
            this.selectFrom3DMode = false;
            // disable data retrieval until next triangle is picked
            TRIANGLE_pickedIndex = GL_NOTFOUND;
        }
    };

    this.openAll = function () {
        this.treeElem.jstree('open_all');
    };

    this.closeAll = function () {
        this.treeElem.jstree('close_all');
    };

    this.setBrainColors = function (colorsUrlList) {

        colorsUrlList = $.parseJSON(colorsUrlList);
        var dataFromServer = [];
        for (var i = 0; i < colorsUrlList.length; i++) {
            var oneData = HLPR_readJSONfromFile(colorsUrlList[i]);
            dataFromServer.push(oneData);
        }
        BASE_PICK_updateBrainColors(dataFromServer);
    };

    this._prepareRegionToTriangleMapping = function () {

        for	(var i = 0; i < this.triaglesMappings.length; i++) {
            var region = this.triaglesMappings[i];
            if (this.regionToTriangleMapping[region]){
                continue;
            }
            this.regionToTriangleMapping[region] = i;
        }
    };

    this._init(baseUrl, treeDataUrl, triangleToRegionUrl, minValue, maxValue);
}