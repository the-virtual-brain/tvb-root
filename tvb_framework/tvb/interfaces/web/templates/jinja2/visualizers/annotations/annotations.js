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

/**
 * Depends on the following GLOBALS: gl, BASE_PICK, ColSch, LEG, HLPR_readJSONfromFile
 *
 * @param treeDataUrl - URL from where the Annotations tree will be loaded
 * @param triangleToRegionUrl - URL for reading a vector of triangle to connectivity region mapping
 * @param activationPatternsUrl - URL for retrieving the Map of the activation patterns
 * @param minValue - Minimum value for the colors (used for legend)
 * @param maxValue - Maximum value for the colors (used for legend)
 * @constructor
 */
function ANN_Displayer(treeDataUrl, triangleToRegionUrl, activationPatternsUrl, minValue, maxValue) {

    this.treeElem = $("#treeStructure");

    this.triaglesMappings = [];
    this.regionToTriangleMapping = {};
    this.selectFrom3DMode = false;

    this.activationPatternMap = {};
    this.regionMappingColors = [];

    this.prefixNodeIdTVB = "node_tvb_";
    this.prefixNodeIdTVBRoot = "node_tvb_root_";
    this.prefixNodeIdBRCO = "node_brco_";

    this._init = function (treeDataUrl, triangleToRegionUrl, activationPatternsUrl, minValue, maxValue) {

        this._populateAnnotationsTree(treeDataUrl);

        ColSch_initColorSchemeGUI(minValue, maxValue);
        LEG_initMinMax(minValue, maxValue);
        LEG_generateLegendBuffers();
        BASE_PICK_initLegendInfo(maxValue, minValue);

        this.triaglesMappings = HLPR_readJSONfromFile(triangleToRegionUrl);
        this._prepareRegionToTriangleMapping();

        this.activationPatternMap = HLPR_readJSONfromFile(activationPatternsUrl);
    };

    this._populateAnnotationsTree = function (treeDataUrl) {

        this.treeElem.jstree({
            "plugins": ["themes", "json_data", "ui", "crrm"],
            "themes": {
                "theme": "default",
                "dots": true,
                "icons": true,
                "url": deploy_context + "/static/jquery/jstree-theme/style.css"
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

            BASE_PICK_clearFocalPoints();
            SELF._redrawColors([]);
            if (SELF.selectFrom3DMode) {
                return;
            }

            BASE_PICK_moveNavigatorToCenter();
            var selectedNode = data.rslt.obj.attr("id");
            if (selectedNode && selectedNode.indexOf(SELF.prefixNodeIdTVB) == 0) {
                selectedNode = selectedNode.replace(SELF.prefixNodeIdTVBRoot, '').replace(SELF.prefixNodeIdTVB, '');
                selectedNode = parseInt(selectedNode);
                displayMessage("Selected Region " + selectedNode, 'infoMessage');

                //TRIANGLE_pickedIndex = parseInt(SELF.regionToTriangleMapping[selectedNode]);
                //BASE_PICK_moveBrainNavigator(true);

                SELF._redrawColors([selectedNode]);

            } else if (selectedNode && selectedNode.indexOf(SELF.prefixNodeIdBRCO) == 0) {
                selectedNode = selectedNode.replace(SELF.prefixNodeIdBRCO, '');
                selectedNode = parseInt(selectedNode);

                var matchingTvbRegions = SELF.activationPatternMap[selectedNode];
                //for (var i = 0; i < matchingTvbRegions.length; i++) {
                //    var triangleIdx = parseInt(SELF.regionToTriangleMapping[matchingTvbRegions[i]]);
                //    TRIANGLE_pickedIndex = triangleIdx;
                //    BASE_PICK_moveBrainNavigator(false);
                //    BASE_PICK_addFocalPoint(triangleIdx);
                //}

                displayMessage("BRCO node connected with " + matchingTvbRegions.length +
                    " TVB regions: \n[" + matchingTvbRegions + "]", 'infoMessage');
                SELF._redrawColors(matchingTvbRegions);
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
            var toBeSelectedId = "#" + this.prefixNodeIdTVBRoot + pickedRegion;
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
        this.regionMappingColors = [];
        for (var i = 0; i < colorsUrlList.length; i++) {
            var oneData = HLPR_readJSONfromFile(colorsUrlList[i]);
            this.regionMappingColors.push(oneData);
        }
        this._redrawColors([]);
    };

    this._redrawColors = function (activeRegions) {

        if (!activeRegions || activeRegions.length == 0) {
            BASE_PICK_updateBrainColors(this.regionMappingColors);
            return;
        }

        var currentColors = [];
        for (var i = 0; i < this.regionMappingColors.length; i++) {
            var chunkColors = this.regionMappingColors[i].slice(0);
            for (var j = 0; j < chunkColors.length; j++) {
                var isActive = false;
                for (var k = 0; k < activeRegions.length; k++) {
                    if (activeRegions[k] == chunkColors[j]) {
                        isActive = true;
                    }
                }
                if (!isActive) {
                    chunkColors[j] = -1;
                }
            }
            currentColors.push(chunkColors);
        }
        BASE_PICK_updateBrainColors(currentColors);
    };


    this._prepareRegionToTriangleMapping = function () {

        for (var i = 0; i < this.triaglesMappings.length; i++) {
            var region = this.triaglesMappings[i];
            if (this.regionToTriangleMapping[region]) {
                continue;
            }
            this.regionToTriangleMapping[region] = i;
        }
    };

    this._init(treeDataUrl, triangleToRegionUrl, activationPatternsUrl, minValue, maxValue);
}